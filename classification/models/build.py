# --------------------------------------------------------
# Iwin Transformer refer to Swin Transformer
# --------------------------------------------------------


from .iwin_transformer import IwinTransformer

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm


    if model_type == 'iwin':
        model = IwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.IWIN.PATCH_SIZE,
                                in_chans=config.MODEL.IWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.IWIN.EMBED_DIM,
                                depths=config.MODEL.IWIN.DEPTHS,
                                num_heads=config.MODEL.IWIN.NUM_HEADS,
                                window_size=config.MODEL.IWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.IWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.IWIN.QKV_BIAS,
                                qk_scale=config.MODEL.IWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.IWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.IWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
