# Video Feature Extraction with Video Swin Transformer

## Installation

Please see the [instruction](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/install.md) of the original repo.

## Preparation

### Video List 

Create a video list with each line containing a video path and a dummy label. For example,

```
/PATH/TO/video1.mp4 0
/PATH/TO/video2.mp4 0
...
```

### Checkpoints

Download checkpoints from the orginal repo.

## Usage

```
python tools/extract.py \
    CONFIG \
    CHECKPOINT \
    OUTPUT \
    --cfg-options \
        data.test.ann_file=FILE_LIST \
        [OTHER_OPTIONS] \
    [--dataset DATASET] 
```

This implementation only supports feature extraction with Swin-B pre-trained on Kinetics 400 or 600 and running on a single GPU.
For example, to extract features of [VATEX](https://eric-xw.github.io/vatex-website/index.html) with Swin-B pre-trained on Kinetics 600,

```sh
python tools/extract.py \
    configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py \
    swin_base_patch244_window877_kinetics600_22k.pth \
    vatex.h5 \
    --cfg-options \
        data.test.ann_file=vatex.txt \
        data.test.pipeline.1.window_interval=32 \
        model.test_cfg.max_testing_views=4 \
    --dataset vatex
```

Set `data.test.pipeline.1.window_interval` to adjust the number of frames between two windows. \
Set `model.test_cfg.max_testing_views` to fit your GPU memory size.

The features of all videos are collected in an hdf5 file `OUTPUT`. \
Specify `--dataset` if you need a customed key for mapping to video feature in the hdf5 file. \
You have to implement the key parser in the function `get_key_parser` in [`tools/extract.py`](https://github.com/shinying/Video-Swin-Transformer/blob/d5f54a3dd3bdf5ae4a369d22d4303d2d51887a27/tools/extract.py#L71), which, given a video path, outputs the video feature key. \
The default feature key of a video is its file name without the path and extension.

