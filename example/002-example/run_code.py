import torch
import time
from example_torchcode import Model, get_inputs, get_init_inputs
from example_cudacode import ModelNew

def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return
    
    device = torch.device("cuda")
    
    # 准备输入数据
    inputs = [x.cuda(device=device) for x in get_inputs()]
    init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in get_init_inputs()]

    # 初始化模型
    torch_model = Model(*init_inputs).cuda()
    cuda_model = ModelNew(*init_inputs).cuda()

    torch_model.eval()
    cuda_model.eval()

    print("-------------------- 精度对齐验证 --------------------")
    with torch.no_grad():
        # 预热GPU
        _ = torch_model(*inputs)
        _ = cuda_model(*inputs)
        
        # 正式测试
        output_torch = torch_model(*inputs)
        output_cuda = cuda_model(*inputs)

    # 精度验证
    abs_diff = torch.abs(output_torch - output_cuda)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()
    
    if max_diff < 1e-4 and mean_diff < 1e-5:
        print(f"✅ 精度对齐：最大误差 {max_diff:.6f}，平均误差 {mean_diff:.6f}")
        precision_flag = True
    else:
        print(f"❌ 精度不一致：最大误差 {max_diff:.6f}，平均误差 {mean_diff:.6f}")
        precision_flag = False

    print("\n-------------------- 性能加速比测试 --------------------")
    num_iterations = 100 
    
    # 预热GPU
    for _ in range(10):
        _ = torch_model(*inputs)
        _ = cuda_model(*inputs)
    
    # PyTorch模型计时
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = torch_model(*inputs)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / num_iterations
    
    # 自定义CUDA内核计时
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = cuda_model(*inputs)
    torch.cuda.synchronize()
    cuda_time = (time.time() - start_time) / num_iterations
    
    print(f"PyTorch内置Swish平均执行时间: {torch_time:.6f}秒")
    print(f"自定义CUDA Swish平均执行时间: {cuda_time:.6f}秒")
    speedup = torch_time / cuda_time if cuda_time > 0 else 0
    print(f"加速比 (Speedup): {speedup:.2f}x")
    
    return precision_flag, speedup

if __name__ == "__main__":
    precision_flag, speedup = run_benchmark()