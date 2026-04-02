
# FlashDiT

## Introduction

FlashDiT is built on the [Lightning DiT project](https://github.com/hustvl/LightningDiT). Unlike LightningDiT, FlashDiT uses the interleaved window attention from the [Iwin Transformer](https://github.com/Cominder/Iwin-Transformer) and removes the positional encoding. FlashDiT is more training faster and less memory usage than LightningDiT.

</div>
<div align="center">
<img src="demo_images/demo_samples.png" alt="Visualization">
</div>

## Train 

Follow [detailed tutorial](docs/tutorial.md) for training your own models.

    ```
    bash run_train.sh configs/flashdit_xl_vavae_f16d32.yaml

## Inference

    Run the following command:
    ```
    bash run_fast_inference.sh configs/flashdit_xl_vavae_f16d32.yaml
    ```  

## Evaluation

    ```
    bash run_fid_eval.sh configs/flashdit_xl_vavae_f16d32.yaml
    ```
    It will provide a reference FID score. For the final reported FID score in the publication, you need to use ADM's evaluation code from [here](https://github.com/openai/guided-diffusion/) for standardized testing.

## Results

| Tokenizer | Generation Model | FID | FID cfg |
|:---------:|:----------------|:----:|:---:|
| [VA-VAE](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt) | [FlashDiT-win8-XL-56ep](https://huggingface.co/cominder/flashdit/blob/main/flashdit-xl-win8-imagenet256-56ep.pt) | 5.37 | 2.30 |


## Acknowledgements

This repo is mainly built on [LightningDiT](https://github.com/hustvl/LightningDiT). Thanks for the great work.


## Citation

If you find the work useful, please cite our related paper:

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
```---
license: mit
---
