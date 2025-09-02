import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 更简单的实现：只优化ReLU部分，矩阵乘法使用PyTorch
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = fmaxf(x[idx], 0.f);
    }
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    return y;
}
"""

relu_cpp_source = """
torch::Tensor relu_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
relu = load_inline(
    name="relu",
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=True
)

class ModelNew(torch.nn.Module):
    def __init__(self, weight):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(weight)
        self.relu = relu  # The module containing the kernel

    def forward(self, x):
        # 使用PyTorch的矩阵乘法，只优化ReLU部分
        x = torch.matmul(x, self.weight)
        return self.relu.relu_cuda(x)
