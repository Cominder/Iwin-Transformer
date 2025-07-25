# model settings
_base_ = "iwin_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2]))
