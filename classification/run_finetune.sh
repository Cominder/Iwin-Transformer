python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window12_384_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 64
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_small_patch4_window12_384_finetune.yaml --pretrained ./output/iwin_small_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 64
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_small_patch4_window16_512_finetune.yaml --pretrained ./output/iwin_small_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 64
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_small_patch4_window16_1024_finetune.yaml --pretrained ./output/iwin_small_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 16



python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window12_384_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 64
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window16_512_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 32
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window16_512_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 32
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window16_1024_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224/default/ckpt_epoch_299.pth --data-path /data/imagenet  --batch-size 8 --accumulation-steps 4



python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin/iwin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224_22k/default/ckpt_epoch_89.pth --data-path /data/imagenet --batch-size 128


python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window12_384_22kto1k_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224_22k/default/ckpt_epoch_89.pth --data-path /data/imagenet  --batch-size 32
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window16_512_22kto1k_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224_22k/default/ckpt_epoch_89.pth --data-path /data/imagenet  --batch-size 32
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_base_patch4_window16_1024_22kto1k_finetune.yaml --pretrained ./output/iwin_base_patch4_window7_224_22k/default/ckpt_epoch_89.pth --data-path /data/imagenet  --batch-size 8 --accumulation-steps 4
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_large_patch4_window7_224_22kto1k_finetune.yaml --pretrained ./output/iwin_large_patch4_window7_224_22k/default/ckpt_epoch_89.pth --data-path /data/imagenet  --batch-size 32
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/iwin_large_patch4_window12_384_22kto1k_finetune.yaml --pretrained ./output/iwin_large_patch4_window7_224_22k/default/ckpt_epoch_89.pth --data-path /data/imagenet  --batch-size 32