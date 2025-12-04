# GPU Emulator

[公众号链接](https://mp.weixin.qq.com/s/znazVk1izwz5i7OIQuaswA?scene=1)

在深度学习、科学计算、高性能计算等领域飞速发展的今天，GPU编程已成为实现高效并行计算的关键。开发者通过CUDA、PyTorch等工具编写kernel，但商业GPU的闭源特性，让kernel的实际执行过程难以观测，架构创新也面临诸多限制。GPU模拟器的出现，为这些问题提供了解决方案。

## 1 为什么需要GPU模拟器

商业GPU的微架构设计与内部优化策略不公开，给开发者和研究者带来两大核心痛点：

* 优化效果难量化：修改调度策略、缓存层次或内存控制逻辑后，无法在真实硬件上精准评估性能变化。
* 创新测试受限制：实验性架构设计或新型硬件改进，难以在真实GPU上开展验证。

而GPU模拟器作为CPU上运行的软件工具，恰好弥补了这些短板：

* 可视化执行过程：追踪每个线程的运行轨迹，清晰呈现内存访问模式、寄存器使用情况和调度逻辑。
* 量化优化效果：在闭源硬件环境中，快速测试新设计方案，精准衡量性能改动。
* 支撑多元场景：既是学术研究的实验平台，也是帮助软件开发者理解GPU并行计算模型的工具。

## 2 GPU模拟器是什么

GPU模拟器的核心是通过软件建模，还原GPU的微架构和执行行为。以主流的GPGPU-Sim为例，其完整工作流程如下：

* 程序输入：接收编译后的二进制程序（如CUDA程序编译产物）。
* 微架构建模：全面模拟GPU核心、SIMT warp执行模型、调度器、寄存器文件、缓存层次、内存控制器等关键组件。现代进阶模拟器（如AccelWattch）还支持功耗建模。
* 性能输出与分析：以cycle-level精度记录GPU核心状态，统计吞吐量、缓存命中率、warp延迟等核心指标，部分模拟器还提供AerialVision等可视化工具，助力定位线程执行瓶颈和内存访问问题。

## 3 怎么本地部署(以GPGPU-SIM为例)

GPGPU-Sim仅支持Linux平台（32/64位均可），部署过程分为环境依赖配置、编译构建、运行验证三步，操作简洁高效。

### 3-1 环境依赖

需提前配置CUDA Toolkit及其他辅助依赖包，具体命令如下：

```Bash
# 安装CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 配置CUDA环境变量
export CUDA_INSTALL_PATH=/usr/local/cuda-11.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 安装其他依赖包
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev doxygen graphviz python-pmw python-ply python-numpy libpng12-dev python-matplotlib libxi-dev libxmu-dev libglut3-dev
```

### 3-2 构建

通过Git克隆源码后，执行简单命令即可完成编译，支持调试和Release两种模式：

```Plain
# 克隆源码仓库
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
cd gpgpu-sim_distribution

# 配置环境（默认release模式，加debug参数可启用调试模式）
source setup_environment  # 或 source setup_environment debug

# 编译构建
make
```

注：Release模式模拟速度更快；调试模式需配合gdb，用于修改模拟器源码后的测试。

### 3-3 运行验证（以向量加法CUDA程序为例）

#### 3-3-1 编写CUDA测试程序(vector_add.cu)

```C++
/* file: vector_add.cu */
#include<stdio.h>#define CHECK(call) \{ \
    const cudaError_t error = call; \if (error != cudaSuccess) { \printf("Error: %s:%d, ", __FILE__, __LINE__); \printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \exit(1); \} \}

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;if (idx < n)
        c[idx] = a[idx] + b[idx];}

int main() {
    int n = 16;
    size_t bytes = n * sizeof(float);

    float h_a[16], h_b[16], h_c[16];for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;}

    float *d_a, *d_b, *d_c;

    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    vector_add<<<1, 16>>>(d_a, d_b, d_c, n);

    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));printf("Result: ");for (int i = 0; i < n; i++) {printf("%.1f ", h_c[i]);}printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);return0;}
```

#### 3-3-2 编译CUDA程序

确保链接GPGPU-Sim的cudart库，编译命令如下：

```Plain
nvcc --cudart shared -o vector_add vector_add.cu
```

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWQ5MjkwMzc5MDY5Zjg1Y2Q3YjQwNjAxNzRlZDIwMWNfR1ZhTEZkTllTV3ExMlZqc29NeDI5Zk1pRWNMRHdvRGNfVG9rZW46VWRzeWJiZTY3b2hOdUl4clZHSGNxRlZIbkNlXzE3NjM0MjU5MzM6MTc2MzQyOTUzM19WNA)

#### 3-3-3 配置模拟架构并运行

从GPGPU-Sim的预定义配置中选择目标架构（以SM86_RTX3070为例），复制配置文件后运行程序：

```Plain
# 复制架构配置文件到当前目录
cp /path/to/gpgpu-sim_distribution/configs/tested-cfgs/SM86_RTX3070/* . -r

# 运行模拟程序
./vector_add
```

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=MmJlZTVlY2Q1YjU4NTVhMzRmMTc2NTY5YjY4NDM4NzdfQlNBUTVnNm1ja2ROc21rQmZXT3lzdG82Q3BkOWpDeG9fVG9rZW46VU95WmJmSlJsb2VOeVh4OUo5M2NZZTVkblNlXzE3NjM0MjU5MzM6MTc2MzQyOTUzM19WNA)

执行后会输出模拟耗时、指令执行速率、周期速率等基础信息，同时打印计算结果（如 `Result: 0.0 3.0 6.0 ... 45.0`）。此外，还会生成详细日志，包含模拟架构参数（SM数量、Warp Scheduler配置、Tensor Core数量等）和kernel执行细节（寄存器/共享内存使用、warp调度、流水线阻塞原因等）。

## 4 GPU模拟器能做哪些分析

GPU模拟器的价值不仅在于“运行”GPU程序，更在于“看透”运行过程、“验证”优化方案，主要体现在两大维度：

### 4-1 解析kernel的GPU执行过程

GPU采用SIMT（单指令多线程）模型，成百上千个线程通过warp协同执行，受限于共享内存、寄存器和缓存，执行逻辑远复杂于CPU的顺序执行。模拟器可解答以下关键问题：

* 线程与warp调度：warp的调度时机、等待资源（内存、执行单元）导致的阻塞原因。
* 存储资源使用：kernel占用的寄存器、共享内存容量，是否存在寄存器溢出或共享内存竞争。
* 内存访问特征：全局/共享/常量内存的访问次数与延迟，缓存命中率及访存瓶颈位置。
* 性能计数器统计：指令总数、浮点运算量、分支指令占比、warp分化情况等核心指标。

例如，运行GEMM（矩阵乘法）kernel时，可通过模拟器观察每个warp的矩阵计算进度、共享内存/寄存器的缓存数据，快速定位访存瓶颈，为优化提供精准依据。

### 4-2 支撑GPU架构优化与创新

模拟器为架构研究者提供了安全可控的实验环境，无需依赖真实硬件即可验证创新设计：

* 微架构参数调优：调整SM数量、warp大小、寄存器文件容量、缓存大小、内存带宽等参数，量化对性能的影响。
* 调度策略分析：对比不同warp调度算法的吞吐量表现，优化内存调度以提升DRAM/L2缓存命中率。
* 新特性验证：模拟Tensor Core等新特性，评估其对整体性能的提升效果。

典型案例：Accel-Sim框架通过模拟Volta架构GPU，发现L1缓存已非性能瓶颈，而内存调度策略对吞吐量影响显著，为后续架构优化指明了方向。

## 5 总结

GPU 模拟器为现代并行计算研究提供了不可或缺的工具。通过前面几节内容，我们可以总结出 GPU 模拟器的核心价值与应用场景：

* 理解 GPU 内部执行：模拟器可以在 cycle-level 精度下追踪 kernel 的线程执行、寄存器与共享内存使用、内存访问模式、warp 调度和 pipeline 行为，让程序员和研究者可以深入观察 GPU 内部运行机制。这对于性能调优和瓶颈分析都有重要意义。
* 探索和验证架构优化：在闭源 GPU 硬件无法直接验证设计改动的情况下，模拟器提供了安全可控的实验平台。研究人员可以调整微架构参数、修改调度策略、验证新硬件特性（如 Tensor Core），量化优化效果，从而推动 GPU 架构创新。
