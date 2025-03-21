# CPU Parallel Computing

> 摩尔定律说每隔18个月，半导体密度都会翻一番，这在过去是正确的。因此硬件的性能在飞速发展，同时其功耗也在不断变大。
>
> 为了减少功耗，增加处理器核是一个不错的选择。当处理器核心增加时，任务可以分配到多个核心上并行执行，这意味着每个核心不需要以很高的主频运行，就能达到和原来单核处理器一样的性能。因为功耗和主频的三次方成正比，故而主频的减少，会带来功耗的减少。
>
> 当处理器由单核变成多核时，程序的编程方法也应该从并发(多个程序逻辑上同时执行)变成并行(多个程序实际上同时执行)，即从concurrency到parallelism：
>
> * concurrency是指，对于单核而言，多个程序通过操作系统的调度，通过上下文切换实现交替执行；
> * parallelism是指，对于多核而言，多个程序可以同时执行，单个程序也可以将其内部任务分配到多个核上并行执行。

最初的多核CPU是将多个CPU核集成到一个芯片上，即采用共享内存模型；而随着计算需求的复杂化和多核处理器规模的扩大，共享内存模型的内存一致性、缓存一致性代价和扩展性不强的特点暴露出来，故而发展了消息传递模型，在该模型中，核心拥有自己独立的内存，数据的交换通过消息的发送和接受来实现。

故而CPU的并行计算可以根据共享内存模型和消息传递模型而分为两类，分别可以对应OpenMP和MPI。

## 1 OpenMP

**Open Multi-Processing**

OpenMP就是一个API用来编写多线程并行程序，即支持单个程序将其内部任务分配到多个核上并行执行。其为写并行程序的程序员提供了一组compiler directives和library routines，支持C/C++/Fortran。使用OpenMP需要程序可并行化、编译器支持OpenMP、操作系统支持共享内存和多线程。因为OpenMP是一种基于共享内存的并行编程模型，所以允许多个线程访问同一块内存空间。(共享内存分为SMP和NUMA，但只要有多级cache，那么就是NUMA，但我们认为是SMP，详见[这里](openmp/pic/SMPandNUMA.jpg))。

### 1-1 头文件

在使用OpenMP的程序中，需要包含头文件omp.h。

```
#include<omp.h>
```

### 1-2 Compiler directives

编译器指令是编程语言中指导编译器行为的特殊指令，比如条件编译 `#ifdef`和 `#endif`可以根据是否定义了某个宏来决定编译器是否编译某段代码。OpenMP中的编译器指令往往是 `#pragma omp parralel [clause [clause ] ...]`，用来进行代码优化。

* #pragma omp parallel {}
* #pragma omp parallel num_threads(n) {}
* #pragma omp barrier
* #pragma omp critical
* #pragma omp atomic
* #pragma omp parallel for {}
* #pragma omp parallel for collapse(N) {}
* #pragma omp parallel for reduction(op:list) {}

OpenMP会让程序从主线程spawn程序员指定个数个线程，然后并行执行某个代码块，即每个线程都执行相同的代码块，当所有线程执行结束后，才会执行代码块后的代码。值得额外说明的是，critical比atomic的粒度要大，开销要大；reduction的操作说明详见[这里](openmp/pic/reduction.jpg)；因为OpenMP是在共享内存的机器上，所以需要考虑[false sharing](openmp/pic/falsesharing.jpg)的情况出现。

### 1-3 Library routines

库例程是预定义在库中的函数，供开发者调用以完成特定任务。

* omp_get_thread_num()
* omp_get_num_threads()
* omp_set_num_threads(NUM_THREADS)

### 1-4 例子

[example_openmp.c](openmp/code/example_openmp.c)

因为gcc支持OpenMP，所以可以通过 `gcc -fopenmp -o example example.c`来编译，而后运行。

### 1-5参考资料

[A &#34;Hands-on&#34; Introduction to OpenMP](openmp/learn/Intro_To_OpenMP_Mattson.pdf)

## 2 MPI

**Message Passing Interface**

MPI是一个API，其有很多实现，这里记录OpenMPI和MPICH。

### 2-1 OpenMPI

#### 2-1-1 安装

```shell
# 从源码编译安装
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz
tar -xvf openmpi-5.0.7.tar.gz
cd openmpi-5.0.7
./configure --prefix=/home/$(whoami)/my-install/openmpi-install
make
make install
# 环境变量配置
export PATH=/home/$(whoami)/my-install/openmpi-install/bin:$PATH
export LD_LIBRARY_PATH=/home/$(whoami)/my-install/openmpi-install/lib:$LD_LIBRARY_PATH
export INCLUDE=/home/$(whoami)/my-install/openmpi-install/include:$INCLUDE
```

#### 2-1-2 编译与运行

```shell
# 编译
mpicc -o example_openmpi example_openmpi.c
# 运行，注意：-n后的参数不能超过实际的CPU核数量；若example_openmpi.c没有使用mpi，也可以用gcc编译后mpiexec来运行
mpiexec -n <number_of_processes> example_openmpi
```

### 2-2 MPICH

#### 2-2-1 安装

```
# 从源码编译安装
wget https://www.mpich.org/static/downloads/4.3.0/mpich-4.3.0.tar.gz
tar -xvf mpich-4.3.0.tar.gz
cd mpich-4.3.0
./configure --prefix=/home/$(whoami)/my-install/mpich-install --disable-fortran
make
make install
# 环境变量配置
export PATH=/home/$(whoami)/my-install/mpich-install/bin:$PATH
export LD_LIBRARY_PATH=/home/$(whoami)/my-install/mpich-install/lib:$LD_LIBRARY_PATH
export INCLUDE=/home/$(whoami)/my-install/mpich-install/include:$INCLUDE
```

#### 2-2-2 编译与运行

```
# 编译
mpicc -o example_mpich example_mpich.c
# 运行，注意：-n后的参数可以超过实际的CPU核数量；若example_mpich.c没有使用mpi，也可以用gcc编译后mpiexec来运行
mpiexec -n <number_of_processes> ./example_mpich
```

### 2-3 例子

[MPI](MPI/MPI.md)

### 2-4 参考资料

[MPI Hands-On C++](https://education.molssi.org/parallel-programming/04-distributed-examples.html)
