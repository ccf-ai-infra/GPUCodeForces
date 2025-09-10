import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用Swish替代原始的ReLU
        return x * torch.sigmoid(x)  # PyTorch内置Swish实现

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # 不需要特殊初始化输入