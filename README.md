# GPUCodeForces

1. 评测数据集生成挑战赛项目

- **目标：** 建立一个专门用来测试GPU性能的数据集。数据集里的内容（即“样本”）可以从流行的工具和模型库中获取：PyTorch, PaddlePaddle, TensorFlow, Jax, MMCV, Transformers 等。
- **评估规则：** 进入评价数据集数量最多的前 12 名
-**评测资源：**MXC500单卡
- **项目价值：** 形成标准的 GPU 评测数据集和评价方法
- **参赛资格：**
只要你至少成功贡献了1个样本（即你的提交经过审核后并被选入“GPU CodeForces”数据集），就可以参赛。

#### 📤 提交要求及评分规则

- 每位选手按一个数据集（JSON 格式）提交，一个完整的GPU CUDA数据集需要包含以下几个部分：
1. 数据集样本描述: 清晰地阐述问题背景、输入、输出和预期功能。
2. 输入数据生成函数: 用于生成各种规模和特性的输入数据的代码。
3. 标准GT输出生函数 : 用于生成给定输入数据的正确输出，通常是Numpy-CPU实现，或原torch/paddle/tensorflow实现。
4. 性能评估指标: 明确评估CUDA解决方案性能的标准（执行时间、吞吐量、内存带宽） 
-数据集样本示例：
1.样本描述 
● 题目名称: 矩阵乘法
● 背景: 矩阵乘法是科学计算、机器学习和图形学中的基本操作。优化其在GPU上的性能是CUDA编程中的一个核心挑战。
● 任务: 给定两个矩阵A和B，计算它们的乘积C = A * B。
2.输入数据生成函数
编写一个脚本，根据给定的 M,K,N 参数，生成两个随机填充的浮点数矩阵A和B
import numpy as np

def generate_matrix_multiplication_data(M, K, N):
    """
    生成矩阵乘法问题的输入数据。
    Args:
        M, K, N: 矩阵A (M x K) 和矩阵B (K x N) 的维度。
    Returns:
        tuple: (matrix_a, matrix_b)
    """
    matrix_a = np.random.uniform(-100.0, 100.0, (M, K)).astype(np.float32)
    matrix_b = np.random.uniform(-100.0, 100.0, (K, N)).astype(np.float32)
    return matrix_a, matrix_b

# 示例用法
# M, K, N = 1024, 512, 256
# A, B = generate_matrix_multiplication_data(M, K, N)
# print(f"Matrix A shape: {A.shape}")
# print(f"Matrix B shape: {B.shape}")

3.输入数据生成函数
编写一个函数，接收输入的矩阵A和B，使用CPU计算出它们的乘积C
import numpy as np

def cpu_matrix_multiplication(matrix_a, matrix_b):
    """
    使用CPU计算矩阵乘法。
    Args:
        matrix_a: 矩阵A (M x K)
        matrix_b: 矩阵B (K x N)
    Returns:
        numpy.ndarray: 矩阵C (M x N)
    """
    return np.dot(matrix_a, matrix_b)

# 示例用法
# A, B = generate_matrix_multiplication_data(1024, 512, 256)
# C_ref = cpu_matrix_multiplication(A, B)
# print(f"Reference C shape: {C_ref.shape}")

4. 输入数据生成函数
● 主要指标: GPU执行时间 (CUDA Kernel Execution Time)。
● 次要指标: 内存带宽利用率、TFLOPS (如果适用)。
● 评判标准:
● 正确性: CUDA解决方案的输出与标准输出的误差应在可接受的浮点误差范围内（例如，np.allclose 容忍度）。
● 性能: 相同输入规模下，CUDA解决方案的执行时间越短越好。我们将提供基准测试环境和计时工具。
● 计时示例 (伪代码/说明):在CUDA代码中，使用 cudaEvent_t 进行精确计时：
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
// Call your CUDA kernel here
my_cuda_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

*若仍有疑问，请详见《评测数据集生成挑战赛样本和要求说明》。
- **接受数量：**提交经过审核后并被选入数据集的样本总数
- **评分方式：**
  - 标准 GT 输出生成函数（Numpy-CPU / 原始框架实现）：+2 分
  - CUDA 性能评估指标：
    - 执行时间（GPU跑完整个任务的耗时）：+5 分
    - 吞吐量（GPU单位时间内处理数据的量）：+4 分
    - 内存带宽（GPU读写的速度）：+3 分
  - （加分项）提供提示语（prompt）让大模型（LLM）生成对应 CUDA 代码，并且这份生成的代码也能提供上述的性能指标，则该提交样本也能得到对应分数。

#### 🏆 排名机制

1. 按“被选入样本的总数”从高到低排序
2. 若数量相同：
   - 比较所有样本的总分数之和，总分数高者优先
   - 若仍相同，比加分项分数高者优先

---