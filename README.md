# Iwin Transformer

## Introduction

**Iwin Transformer** (the name `Iwin` stands for **I**nterleaved **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2507.18405). It is a position-embedding-free hierarchical vision transformer, which can be fine-tuned directly from low to high resolution, through the collaboration of innovative interleaved window attention and depthwise separable convolution.

![teaser](classification/figures/teaser1.png)
![teaser](classification/figures/teaser2.png)
![teaser](classification/figures/teaser3.png)
![teaser](classification/figures/teaser4.png)

## Results on ImageNet with Pretrained Models

**ImageNet-1K and ImageNet-22K Pretrained Iwin Models**

| name | pretrain | resolution | acc@1 | #params | FLOPs | 22K model | 1K model |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| Iwin-T | ImageNet-1K | 224x224 | 82.0 | 30.2M | 4.7G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_patch4_window7_224.pth)/[config](configs/iwin/iwin_tiny_patch4_window7_224.yaml) |
| Iwin-S | ImageNet-1K | 224x224 | 83.4 | 51.6M | 9.0G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_patch4_window7_224.pth)/[config](configs/iwin/iwin_small_patch4_window7_224.yaml) |
| Iwin-S | ImageNet-1K | 384x384 | 84.3 | 51.6M | 27.7G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_patch4_window12_384.pth)/[config](configs/iwin/iwin_small_patch4_window12_384_finetune.yaml) |
| Iwin-S | ImageNet-1K | 512x512 | 84.4 | 51.6M | 52.0G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_patch4_window16_512.pth)/[config](configs/iwin/iwin_small_patch4_window16_512_finetune.yaml) |
| Iwin-S | ImageNet-1K | 1024x1024 | 83.8 | 51.6M | 207.9G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_patch4_window16_1024.pth)/[config](configs/iwin/iwin_small_patch4_window16_1024_finetune.yaml) |
| Iwin-B | ImageNet-1K | 224x224 | 83.5 | 91.2M | 15.9G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window7_224.pth)/[config](configs/iwin/iwin_base_patch4_window7_224.yaml) |
| Iwin-B | ImageNet-1K | 384x384 | 84.9 | 91.2M | 48.3G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window12_384.pth)/[config](configs/iwin/iwin_base_patch4_window12_384_finetune.yaml) |
| Iwin-B | ImageNet-1K | 512x512 | 85.1 | 91.3M | 89.5G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window16_512.pth)/[config](configs/iwin/iwin_base_patch4_window16_512_finetune.yaml) |
| Iwin-B | ImageNet-1K | 1024x1024 | 85.0 | 91.3M | 358.2G | - | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window16_1024.pth)/[config](configs/iwin/iwin_base_patch4_window16_1024_finetune.yaml) |
| Iwin-B | ImageNet-22K | 224x224 | 85.5 | 91.2M | 15.9G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window7_224_22k.pth)/[config](configs/iwin/iwin_base_patch4_window7_224_22k.yaml) | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window7_224_22kto1k.pth)/[config](configs/iwin/iwin_base_patch4_window7_224_22kto1k_finetune.yaml) |
| Iwin-B | ImageNet-22K | 384x384 | 86.6 | 91.2M | 48.3G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window12_384_22k.pth)/[config](configs/iwin/iwin_base_patch4_window12_384_22k.yaml) | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window12_384_22kto1k.pth)/[config](configs/iwin/iwin_base_patch4_window12_384_22kto1k_finetune.yaml) |
| Iwin-B | ImageNet-22K | 512x512 | 86.1 | 91.2M | 89.5G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window16_512_22k.pth)/[config](configs/iwin/iwin_base_patch4_window16_512_22k.yaml) | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window16_512_22kto1k.pth)/[config](configs/iwin/iwin_base_patch4_window16_512_22kto1k_finetune.yaml) |
| Iwin-B | ImageNet-22K | 1024x1024 | 85.6 | 91.2M | 358.2G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window16_1024_22k.pth)/[config](configs/iwin/iwin_base_patch4_window16_1024_22k.yaml) | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window16_1024_22kto1k.pth)/[config](configs/iwin/iwin_base_patch4_window16_1024_22kto1k_finetune.yaml) |
| Iwin-L | ImageNet-22K | 224x224 | 86.4 | 204.3M | 35.4G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_large_patch4_window7_224_22k.pth)/[config](configs/iwin/iwin_large_patch4_window7_224_22k.yaml) | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_large_patch4_window7_224_22kto1k.pth)/[config](configs/iwin/iwin_large_patch4_window7_224_22kto1k_finetune.yaml) |
| Iwin-L | ImageNet-22K | 384x384 | 87.4 | 204.3M | 106.6G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_large_patch4_window12_384_22k.pth)/[config](configs/iwin/iwin_large_patch4_window12_384_22k.yaml) | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_large_patch4_window12_384_22kto1k.pth)/[config](configs/iwin/iwin_large_patch4_window12_384_22kto1k_finetune.yaml) |



## Results on Downstream Tasks

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


**ADE20K Semantic Segmentation (val)**

| Backbone | Method | pretrain | Crop Size | Lr Schd | mIoU | #params | FLOPs | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Swin-T | UPerNet | ImageNet-1K | 512x512 | 160K | 44.70 | 61.9M | 946G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_patch4_window7_512_ade20k_1k.pth)|
| Swin-S | UperNet | ImageNet-1K | 512x512 | 160K | 47.50 | 83.2M | 1038G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_small_patch4_window7_512_ade20k_1k.pth)|
| Swin-B | UperNet | ImageNet-1K | 512x512 | 160K | 48.90 | 124.8M | 1189G | [github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_base_patch4_window7_512_ade20k_1k.pth)|


**Kinetics 400 Recognition**


### **Kinetics-400 Video Recognition**

| Backbone | Pretrain | Lr Schd | spatial crop | acc@1 | acc@5 | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Iwin-T | ImageNet-1K | 30ep | 224 | 79.1 | 93.8 | 29.8M | 74G | [config](video_recognition/configs/recognition/iwin/iwin_tiny_patch244_window77_kinetics400_1k.py) |[github](https://github.com/Cominder/Iwin-Transformer/releases/download/v1.0/iwin_tiny_patch244_window77_kinetics400_1k.pth) |
| Iwin-S | ImageNet-1K | 30ep | 224 | 80.0 | 94.1 | 51.1M | 140G | [config](video_recognition/configs/recognition/iwin/iwin_small_patch244_window77_kinetics400_1k.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0/iwin_small_patch244_window77_kinetics400_1k.pth) |





## Acknowledgements 

This repo is mainly built on [Swin](https://github.com/microsoft/Swin-Transformer). Thanks for the great works.


## Citation

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


