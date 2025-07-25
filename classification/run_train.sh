#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_tiny_patch4_window7_224.yaml --data-path /data/imagenet --batch-size 512
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_large_patch4_window7_224_22k.yaml --data-path /data/imagenet22k --batch-size 256 --accumulation-steps 2

