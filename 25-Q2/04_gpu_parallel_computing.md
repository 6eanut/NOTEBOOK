# GPU Parallel Computing

并行计算除了CPU之外，还有GPU，这里记录CUDA的基本使用。

## 1 前言

这里使用WSL 2作为开发环境：

```shell
# 安装方式，在powershell中以管理员方式运行
wsl --install
# 查看已安装的发行版
wsl -v -l
# 停止实例
wsl --shutdown
# 导出发行版到D盘
wsl --export Ubuntu-22.04 D:\wsl_backup\ubuntu.tar
# 注销原发行版
wsl --unregister Ubuntu-22.04
# 在D盘创建目录
mkdir D:\WSL\Ubuntu-22.04
# 导入到D盘
wsl --import Ubuntu-22.04 D:\WSL\Ubuntu-22.04 D:\wsl_backup\ubuntu.tar --version 2
# 启动实例
wsl -d Ubuntu-22.04
# 查看GPU信息
nvidia-smi
# 配置cuda环境
sudo apt update
sudo apt install nvidia-cuda-toolkit
# 编写一个cuda程序
vi hello.cu
# 编译并运行
nvcc -o hello hello.cu -run
```

## 2 基本用法

```c
void CPUFunction()
{
  printf("This function is defined to run on the CPU.\n");
}

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.\n");
}

int main()
{
  CPUFunction();

  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```

`__global__ void GPUFunction()`

* `__global__` 关键字表明以下函数将在 GPU 上运行并可**全局**调用，而在此种情况下，则指由 CPU 或 GPU 调用。
* 通常，我们将在 CPU 上执行的代码称为**主机**代码，而将在 GPU 上运行的代码称为**设备**代码。
* 注意返回类型为 `void`。使用 `__global__` 关键字定义的函数需要返回 `void` 类型。

`GPUFunction<<<1, 1>>>();`

* 通常，当调用要在 GPU 上运行的函数时，我们将此种函数称为**已启动**的 **核函数** 。
* 启动核函数时，我们必须提供 **执行配置** ，即在向核函数传递任何预期参数之前使用 `<<< ... >>>` 语法完成的配置。
* 在宏观层面，程序员可通过执行配置为核函数启动指定 **线程层次结构** ，从而定义线程组（称为 **线程块** ）的数量，以及要在每个线程块中执行的**线程**数量。稍后将在本实验深入探讨执行配置，但现在请注意正在使用包含 `1` 线程（第二个配置参数）的 `1` 线程块（第一个执行配置参数）启动核函数。

`cudaDeviceSynchronize();`

* 与许多 C/C++ 代码不同，核函数启动方式为 **异步** ：CPU 代码将继续执行 *而无需等待核函数完成启动* 。
* 调用 CUDA 运行时提供的函数 `cudaDeviceSynchronize` 将导致主机 (CPU) 代码暂作等待，直至设备 (GPU) 代码执行完成，才能在 CPU 上恢复执行。

## 3 线程结构

### 3-1 基本用法

线程的集合称为块，块的集合称为网格，<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>执行配置定义了网格中的块数和块中的线程数。

```c
# 索引从0开始编号
gridDim.x 	->	网格中的块数
blockIdx.x 	->	网格中当前块的索引
blockDim.x	->	块中的线程数
threadIdx.x	->	块中当前线程的索引
```

应用：

```c
#include <stdio.h>

__global__ void printIDX()
{

  if(threadIdx.x == 2 && blockIdx.x == 2)
  {
   printf("HERE\n");
  } else {
   printf("THERE\n");
  }
}

int main()
{
  printIDX<<<3, 3>>>();
  cudaDeviceSynchronize();
}
```

### 3-2 块间协作

通过 `blockDim.x * blockIdx.x + threadIdx.x`可以得到当前线程在整个网格中的索引，这很重要，下面是一个应用：

> 每个块中的线程数的设置有上限，目前是不能超过1024，故而当想要更多线程参与并行计算时，就不得不启用多个块，这就涉及到块间的协作了。

```c
#include <stdio.h>

__global__ void printIDX()
{
  printf("%d\n",threadIdx.x+blockIdx.x*blockDim.x);
}

int main()
{
  printIDX<<<2, 5>>>();
  cudaDeviceSynchronize();
}
```

### 3-3 执行配置的设置套路

> 鉴于GPU的硬件特性，一个块包含的线程数如果是32的倍数，性能会比较好，所以一般会设置成256。
>
> 但这样会出现一个问题，即执行配置分配的线程数和任务需要的线程数不匹配，故而需要在核函数里面筛选一下，当检测到当前线程是多余的时，什么也不做。
>
> 启动核函数时需要指定执行配置，即块数和每个块的线程数，每个块的线程数往往设置成256，且一般已知处理某个任务想要分配的线程数N(可能不是256的整数倍)，故而可以对N向上对齐。

以下是编写执行配置的惯用方法示例，适用于 `N` 和线程块中的线程数已知，但无法保证网格中的线程数和 `N` 之间完全匹配的情况。如此一来，便可确保网格中至少始终拥有 `N` 所需的线程数，且超出的线程数至多仅可相当于 1 个线程块的线程数量：

```cpp
// Assume `N` is known
int N = 100000;

// Assume we have a desire to set `threads_per_block` exactly to `256`
size_t threads_per_block = 256;

// Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

some_kernel<<<number_of_blocks, threads_per_block>>>(N);
```

由于上述执行配置致使网格中的线程数超过 `N`，因此需要注意 `some_kernel` 定义中的内容，以确保 `some_kernel` 在由其中一个 ”额外的” 线程执行时不会尝试访问超出范围的数据元素：

```cpp
__global__ some_kernel(int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N) // Check to make sure `idx` maps to some value within `N`
  {
    // Only do work if it does
  }
}
```

### 3-4 小问题

在3-3中提到执行任务所需要的线程数和分配的线程数不匹配，这涉及到两种情况，即前者大于后者和前者小于后者，在3-3中只解决了前者小于后者的情况，这里讨论前者大于后者的情况。

CUDA 提供一个可给出网格中线程块数的特殊变量：`gridDim.x`。然后计算网格中的总线程数，即网格中的线程块数乘以每个线程块中的线程数：`gridDim.x * blockDim.x`。带着这样的想法来看看以下核函数中网格跨度循环的详细示例：

```cpp
__global void kernel(int *a, int N)
{
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;

  for (int i = indexWithinTheGrid; i < N; i += gridStride)
  {
    // do work on a[i];
  }
}
```

## 4 内存管理

### 4-1 UM分配和释放

CPU代码中malloc的内存并不能在GPU代码中访问，要想管理GPU的内存，需要用到特定的函数。

如要分配和释放内存，并获取可在主机和设备代码中引用的指针，请使用 `cudaMallocManaged` 和 `cudaFree` 取代对 `malloc` 和 `free` 的调用，如下例所示：

```cpp
// CPU-only

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
a = (int *)malloc(size);

// Use `a` in CPU-only program.

free(a);
```

```cpp
// Accelerated

int N = 2<<20;
size_t size = N * sizeof(int);

int *a;
// Note the address of `a` is passed as first argument.
cudaMallocManaged(&a, size);

// Use `a` on the CPU and/or on any GPU in the accelerated system.

cudaFree(a);
```

### 4-2 内存异步预取

cudaMallocManaged函数申请的内存，在程序真正访问内存之前，并不会实际分配，当真正访问时，会发生缺页异常而后再分配，这个过程比较影响性能，所以我们可以异步预取。

如果整个程序只在CPU代码内或只在GPU代码内访问申请的内存，则不会发生内存迁移。

CUDA 可通过 `cudaMemPrefetchAsync` 函数，轻松将托管内存异步预取到 GPU 设备或 CPU。以下所示为如何使用该函数将数据预取到当前处于活动状态的 GPU 设备，然后再预取到 CPU：

```cpp
int deviceId;
cudaGetDevice(&deviceId);                                         // The ID of the currently active GPU device.

cudaMemPrefetchAsync(pointerToSomeUMData, size, deviceId);        // Prefetch to GPU device.
cudaMemPrefetchAsync(pointerToSomeUMData, size, cudaCpuDeviceId); // Prefetch to host. `cudaCpuDeviceId` is a
                                                                  // built-in CUDA variable.
```


## 5 GPU硬件信息

一个GPU包含多个SM(Streaming Multiprocessors)，但具体包含多少个呢？这可以通过函数来获取，比如版本号，SM个数，Warp数量：

```c
#include<stdio.h>
int main(){
    int deviceId;
    cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.
  
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about
                                               // the active GPU device.
  
    int computeCapabilityMajor = props.major;  // 版本号
    int computeCapabilityMinor = props.minor;  // 版本号
    int multiProcessorCount = props.multiProcessorCount;  // SM数量
    int warpSize = props.warpSize;  // Warp数量，通常是32，GPU调度的基本线程单元
  
    printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
```

拿到SM个数有啥用呢？在核函数执行期间，线程块会被分配到SM上来执行，GPU调度的基本单元是32个线程的warp，故而 线程块数是SM数的整数倍 和 块中线程数是warp的整数倍 有利于性能提升。

## 小结

附：错误处理见[这里](cuda/handleerror.md)，nsys使用方法见[这里](cuda/nsys.md)，流相关概念见[这里](cuda/stream.md)。

以下是模板：

```c
#include <stdio.h>

void init(float* a, int N)
{
  for(int i = 0; i < N; i++)
    a[i] = 0;
}

// 1 -> __global__
__global__ void deviceCode(float* a, int N)
{

// 2 -> indexWithinTheGrid, gridStride, for
  int indexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int gridStride = gridDim.x * blockDim.x;
  for (int i = indexWithinTheGrid; i < N; i += gridStride)
    a[i] += 1;
}

void check(float* a, int N)
{
  for(int i = 0; i < N; i++)
    if(a[i] != 1){
      printf("error\n");
      return;
    }
  printf("right\n");
}

int main()
{
// 3 -> deviceId, SM
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  int multiProcessorCount = props.multiProcessorCount;

  const int N = 2<<20;
  size_t size = N * sizeof(float);
  float *a;

// 4 -> cudaMallocManaged
  cudaMallocManaged(&a, size);

// 5 -> TO HOST
  cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);

  init(a, N);

// 6 -> HOST TO DEVICE
  cudaMemPrefetchAsync(a, size, deviceId);

// 7 -> threads_per_block, number_of_blocks
  size_t threads_per_block = 256;
  //size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
  size_t number_of_blocks = multiProcessorCount * 32; // better

// 8 -> executation configuration
  deviceCode<<<number_of_blocks,threads_per_block>>>(a, N);

// 9 -> cudaDeviceSynchronize
  cudaDeviceSynchronize();

// 10 -> DEVICE TO HOST, 一定要在同步之后
  cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);

  check(a, N);

// 11 -> cudaFree
  cudaFree(a);
}
```
