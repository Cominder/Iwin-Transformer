import torch
import torch.nn as nn

depthwise = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32).cuda()
x = torch.randn(4, 32, 64, 64).cuda()

output = depthwise(x)  # 确认是否有设备不一致的问题

