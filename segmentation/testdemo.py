from mmseg.apis import inference_segmentor, init_segmentor
 
config_file = 'configs/iwin/upernet_iwin_tiny_patch4_window7_512x512_160k_ade20k.py'  
checkpoint_file = './upernet_iwin_tiny_patch4_window7_512x512.pth'                                    
device = 'cuda:0'

img = 'demo/demo.png'

model = init_segmentor(config_file, checkpoint_file, device=device)
# test a single image
result = inference_segmentor(model, img)
# save the results
model.show_result(img, result, out_file='demo_result.jpg')                 
