import torch
import torch.nn as nn

class NQLoss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, x, y):  # 定义前向的函数运算即可
        return torch.mean(torch.pow((x - y), 2))