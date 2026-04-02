# bash tools/dist_train.sh configs/recognition/iwin/iwin_tiny_patch244_window877_kinetics400_1k.py 8 --cfg-options model.backbone.pretrained=./iwin_tiny.pth
bash tools/dist_train.sh configs/recognition/iwin/iwin_small_patch244_window877_kinetics400_1k.py 8 --cfg-options model.backbone.pretrained=./iwin_small.pth
bash tools/dist_train.sh configs/recognition/iwin/iwin_base_patch244_window877_kinetics400_22k.py 8 --cfg-options model.backbone.pretrained=./iwin_base_22k.pth



bash tools/dist_train.sh configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py 8 --cfg-options model.backbone.pretrained=./swin_tiny.pth

