import torch
from torch.utils.cpp_extension import load_inline

# Swish激活函数的CUDA实现 (x * sigmoid(x))
swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 高效计算Swish: x * (1 / (1 + exp(-x)))
        float val = x[idx];
        float sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid;
    }
}

torch::Tensor swish_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    swish_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    return y;
}
"""

swish_cpp_source = """
torch::Tensor swish_cuda(torch::Tensor x);
"""

# 编译内联CUDA代码
swish = load_inline(
    name="swish",
    cpp_sources=swish_cpp_source,
    cuda_sources=swish_source,
    functions=["swish_cuda"],
    verbose=True
)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.swish = swish  # 包含自定义Swish算子的模块

    def forward(self, x):
        return self.swish.swish_cuda(x)