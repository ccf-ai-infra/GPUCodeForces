###########################################################
# 性能和精度验证程序
###########################################################
import torch
import torch.nn as nn
import time
from example_torchcode import Model,get_inputs,get_init_inputs
from example_cudacode import ModelNew

def run_benchmark():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用，请确保您有可用的 NVIDIA GPU 并已正确安装 PyTorch CUDA 版本。")
        return
    else:
        device = torch.device("cuda")

    # 初始化模型
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]
    inputs = get_inputs()
    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in inputs
    ]

    torch_model = Model(*init_inputs).cuda()
    cuda_model = ModelNew(*init_inputs).cuda()

    torch_model.eval()
    cuda_model.eval()

    print("-------------------- 精度对齐验证 --------------------")
    with torch.no_grad():
        output_torch = torch_model( *inputs)
        output_cuda = cuda_model(*inputs)

    precision_flag = torch.allclose(output_torch, output_cuda,rtol=1e-03)
    if precision_flag:
        print("✅ 精度对齐：两个模型的输出结果非常接近。")
    else:
        print("❌ 精度不一致！")
        
    print("\n-------------------- 性能加速比测试 --------------------")
    num_iterations = 100 
    
    # PyTorch 模型计时
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = torch_model(*inputs)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / num_iterations
    
    # 自定义 CUDA 内核计时
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = cuda_model(*inputs)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / num_iterations
    
    print(f"PyTorch torch.relu 平均执行时间: {torch_time:.6f} 秒")
    print(f"自定义 CUDA 内核 平均执行时间: {cuda_time:.6f} 秒")
    speedup = 0
    if cuda_time > 0:
        speedup = torch_time / cuda_time
        print(f"加速比 (Speedup): {speedup:.2f}x")
    else:
        print("CUDA 内核执行时间为0，无法计算加速比。")
    return precision_flag,speedup
if __name__ == "__main__":
    precision_flag,speedup = run_benchmark()