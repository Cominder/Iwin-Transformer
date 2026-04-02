pred_npz=$1

python guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz \
    ${pred_npz}