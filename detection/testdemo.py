from mmdet.apis import init_detector, inference_detector
 
config_file = 'configs/iwin/mask_rcnn_iwin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py' 
checkpoint_file = './checkpoints/iwin_tiny_patch4_window7_mask_rcnn_1x.pth'                                     
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img = 'demo/demo.jpg'
result = inference_detector(model, 'demo/demo.jpg')
model.show_result(img, result)
model.show_result(img, result, out_file='demo/demo_result.jpg')
print('saved')
