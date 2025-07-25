# #bash tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=swin_tiny_patch4_window7_224.pth
# #bash tools/dist_train.sh configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# #bash tools/dist_train.sh configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=ckpt_epoch_299.pth
# #bash tools/dist_train.sh configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth

# #bash tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 --cfg-options model.pretrained=swin_tiny_patch4_window7_224.pth
# #bash tools/dist_train.sh configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth model.backbone.use_checkpoint=True
# # bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_small_patch4_window7_224.pth
# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_small_patch4_window7_224.pth

# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_small_patch4_window7_224.pth


# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# #bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patch4_window7_224.pth
# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_small_patch4_window7_224.pth

# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_small_patch4_window7_224.pth

# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_small_patch4_window7_224.pth
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# bash tools/dist_train.sh  configs/swinfusion/mask_rcnn_swin_fusion_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=swin_fusion_tiny.pth
# bash tools/dist_train.sh  configs/swinfusion/mask_rcnn_swin_fusion_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=swin_fusion_small.pth

# bash tools/dist_train.sh  configs/iwinfusion/mask_rcnn_iwin_fusion_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_fusion_tiny.pth
# bash tools/dist_train.sh  configs/iwinfusion/mask_rcnn_iwin_fusion_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_fusion_small.pth

# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_small.pth

# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options --resume-from /data/detection/work_dirs/epoch_28.pth

bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --resume-from /data/detection/epoch_15.pth

# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny_patchmerging.pth
# bash tools/dist_train.sh  configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_tiny.pth
# bash tools/dist_train.sh  configs/iwin/reppoitsv2_iwin_tiny_patch4_window7_mstrain_480_960_giou_gfocal_bifpn_adamw_3x_coco.py 8 --cfg-options model.pretrained=iwin_tiny.pth

# bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --resume-from epoch_24.pth
 bash tools/dist_train.sh  configs/iwin/cascade_mask_rcnn_iwin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py 8 --cfg-options model.pretrained=iwin_small.pth