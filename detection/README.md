# Iwin Transformer for Object Detection

## Results and Models

**COCO Object Detection (2017 val)**

| Backbone | Method | pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | model
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Iwin-T | Mask R-CNN | ImageNet-1K | 1x | 42.2 | 38.9 | 48M | 268G |  [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_window7_mask_rcnn_1x_coco.pth)|
| Iwin-S | Mask R-CNN | ImageNet-1K | 1x | 43.7 | 40.0 | 69M | 358G |  [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_window7_mask_rcnn_1x_coco.pth)|
| Iwin-T | Mask R-CNN | ImageNet-1K | 3x | 44.7 | 40.9 | 48M | 268G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_window7_mask_rcnn_3x_coco.pth)|
| Iwin-S | Mask R-CNN | ImageNet-1K | 3x | 45.5 | 41.0 | 69M | 358G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_window7_mask_rcnn_3x_coco.pth)|
| Iwin-T | Cascade Mask R-CNN | ImageNet-1K | 1x | 47.2 | 40.9 | 86M | 747G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_window7_cascade_mask_rcnn_1x_coco.pth)|
| Iwin-T | Cascade Mask R-CNN | ImageNet-1K | 3x | 49.4 | 42.9 | 86M | 747G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_window7_cascade_mask_rcnn_3x_coco.pth)|
| Iwin-S | Cascade Mask R-CNN | ImageNet-1K | 3x | 49.4 | 43.0 | 107M | 837G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_window7_cascade_mask_rcnn_3x_coco.pth)|



## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Iwin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/iwin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citing Iwin Transformer
```
@misc{huo2025iwin,
      title={Iwin Transformer: Hierarchical Vision Transformer using Interleaved Windows}, 
      author={Simin Huo and Ning Li},
      year={2025},
      eprint={2507.18405},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.18405}, 
}
```

## Other Links

> **Image Classification**: See [Iwin Transformer for Image Classification](https://github.com/Cominder/Iwin-Transformer/classification).

> **Semantic Segmentation**: See [Iwin Transformer for Semantic Segmentation](https://github.com/Cominder/Iwin-Transformer/segmentation).

> **Video Recognition**: See [Iwin Transformer for Object Detection](https://github.com/Cominder/Iwin-Transformer/video_recognition).

