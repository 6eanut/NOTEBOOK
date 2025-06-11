## 并发CUDA流

先浏览该[文件](NVVP-Streams-1-zh.pdf)。

在 CUDA 编程中，**流**是由按顺序执行的一系列命令构成。在 CUDA 应用程序中，核函数的执行以及一些内存传输均在 CUDA 流中进行。不过直至此时，您仍未直接与 CUDA 流打交道；但实际上您的 CUDA 代码已在名为*默认流*的流中执行了其核函数。

除默认流以外，CUDA 程序员还可创建并使用非默认 CUDA 流，此举可支持执行多个操作，例如在不同的流中并发执行多个核函数。多流的使用可以为您的加速应用程序带来另外一个层次的并行，并能提供更多应用程序的优化机会。

### 控制CUDA流行为的规则

为有效利用 CUDA 流，您应了解有关 CUDA 流行为的以下几项规则：

* 给定流中的所有操作会按序执行。
* 就不同非默认流中的操作而言，无法保证其会按彼此之间的任何特定顺序执行。
* 默认流具有阻断能力，即，它会等待其它已在运行的所有流完成当前操作之后才运行，但在其自身运行完毕之前亦会阻碍其它流的运行。

### 创建，使用和销毁非默认CUDA流

以下代码段演示了如何创建，利用和销毁非默认CUDA流。您会注意到，要在非默认CUDA流中启动CUDA核函数，必须将流作为执行配置的第4个可选参数传递给该核函数。到目前为止，您仅利用了执行配置的前两个参数：

```C++
cudaStream_t stream;   // CUDA流的类型为 `cudaStream_t`
cudaStreamCreate(&stream); // 注意，必须将一个指针传递给 `cudaCreateStream`

someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>();   // `stream` 作为第4个EC参数传递

cudaStreamDestroy(stream); // 注意，将值（而不是指针）传递给 `cudaDestroyStream`
```

但值得一提的是，执行配置的第3个可选参数超出了本实验的范围。此参数允许程序员提供 **共享内存** （当前将不涉及的高级主题）中为每个内核启动动态分配的字节数。每个块分配给共享内存的默认字节数为“0”，在本练习的其余部分中，您将传递“ 0”作为该值，以便展示我们感兴趣的第4个参数。

### 练习：将流用于并行进行数据初始化的核函数

您一直使用的向量加法应用程序 [01-prefetch-check-solution.cu](code/01-prefetch-check-solution.cu) 目前启动 3 次初始化核函数，即：为 `vectorAdd` 核函数需要初始化的 3 个向量分别启动一次。重构该应用程序，以便在其各自的非默认流中启动全部 3 个初始化核函数。在使用下方的代码执行单元编译及运行时，您应仍能看到打印的成功消息。如您遇到问题，请参阅 [解决方案](code/solutions/01-stream-init-solution.cu)。

```
nvcc -o init-in-streams 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

在Nsight Systems中打开一个报告，以确认您的3个初始化内核启动正在它们自己的非默认流中运行，并且存在一定程度的并发重叠。

```
nsys profile --stats=true -o init-in-streams-report ./init-in-streams
```
