import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output', help='output feature path')
    parser.add_argument('--dataset', help='dataset', default='')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def get_key_parser(dataset=''):
    default = lambda f: osp.splitext(osp.basename(f))[0]
    if dataset == 'webvid':
        return lambda f: os.sep.join(osp.splitext(f)[0].split(os.sep)[-2:])
    if dataset == 'anetqa':
        return lambda f: osp.splitext(osp.basename(f))[0][2:]
    if len(dataset):
        print(f"Warning: dataset `{dataset}` is not supported. Use basenames without extension as feature keys")
    else:
        print("Warning: No dataset name is given. Use basenames without extension as feature keys")
    return default


def single_gpu_extract(model, data_loader, output, key_parser):
    """Extract features with a single gpu.
    This method extracts features with a single gpu and displays test progress bar.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
    """
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    with h5py.File(output, 'a') as fd:
        for data in data_loader:
            vid_key = key_parser(data['img_metas'].data[0][0]['filename'])
            if vid_key not in fd:
                del data['img_metas']
                with torch.no_grad():
                    result = model(return_loss=False, **data)
                fd.create_dataset(vid_key, data=result.squeeze(0))

            # Assume result has the same length of batch_size
            # refer to https://github.com/open-mmlab/mmcv/issues/985
            prog_bar.update()


def inference_pytorch(args, cfg, distributed):
    """Get predictions by pytorch models.
    """
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # ========== build the model and load checkpoint ========== #

    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # ========== build the dataset ========== #

    key_parser = get_key_parser(args.dataset)

    if osp.isfile(args.output): # resuming from incomplete trial
        vidlist = open(cfg.data.test.ann_file).read().splitlines()
        all_len = len(vidlist)
        with h5py.File(args.output, 'r') as f:
            vidlist = [l for l in vidlist if not key_parser(l.split()[0]) in f]

        print(f'Found {args.output}. Remain {len(vidlist)}/{all_len}')

        filtered_file = f'/tmp/filtered_{osp.basename(cfg.data.test.ann_file)}'
        with open(filtered_file, 'w') as fd:
            for l in vidlist:
                print(l, file=fd)
        cfg.data.test.ann_file = filtered_file

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # ========== extract feature ========== #

    if not distributed:
        # model = MMDataParallel(model, device_ids=[0])
        model = model.cuda()
        outputs = single_gpu_extract(model, data_loader, args.output, key_parser)
    else:
        raise NotImplementedError
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    return outputs


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)
    assert cfg.data.test.ann_file, "No video list provided"
    print("Input file:", cfg.data.test.ann_file)

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    inference_pytorch(args, cfg, distributed)


if __name__ == '__main__':
    main()
