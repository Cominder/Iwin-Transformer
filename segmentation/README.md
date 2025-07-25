# Iwin Transformer for Semantic Segmentaion

## Results and Models

### ADE20K

| Backbone | Method | pretrain | Crop Size | Lr Schd | mIoU | #params | FLOPs | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Iwin-T | UPerNet | ImageNet-1K | 512x512 | 160K | 44.70 | 61.9M | 946G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_patch4_window7_512_ade20k_1k.pth)|
| Iwin-S | UperNet | ImageNet-1K | 512x512 | 160K | 47.50 | 83.2M | 1038G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_patch4_window7_512_ade20k_1k.pth)|
| Iwin-B | UperNet | ImageNet-1K | 512x512 | 160K | 48.90 | 124.8M | 1189G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window7_512_ade20k_1k.pth)|



## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train an UPerNet model with a `Iwin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/iwin/upernet_iwin_tiny_patch4_window7_512x512_160k_ade20k.py 8 --options model.pretrained=<PRETRAIN_MODEL> 
```

**Notes:** 
- `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


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

> **Object Detection**: See [Iwin Transformer for Object Detection](https://github.com/SwinTransformer/Iwin-Transformer/detection).

> **Video Recognition**: See [Iwin Transformer for Object Detection](https://github.com/Cominder/Iwin-Transformer/video_recognition).
