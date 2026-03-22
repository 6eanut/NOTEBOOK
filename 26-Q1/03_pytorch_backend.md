# Integrating a Custom Backend into PyTorch

## 1 什么是 PyTorch？什么是 PyTorch 的 Backend？

**PyTorch** 是一个开源的**深度学习框架和张量计算库** ，主要用于构建和训练神经网络。它提供了 Python 接口，让用户可以方便地定义神经网络、进行张量运算以及自动求导（autograd）。虽然用户主要通过 Python 编写代码，但真正执行计算的是底层的高性能 C++/CUDA 实现，因此 PyTorch 既易用又具有很高的计算效率。

**PyTorch 的 Backend** 指的是**负责实际执行张量运算的底层实现层** 。当用户在 PyTorch 中执行一个操作（例如张量加法或矩阵乘法）时，框架会根据张量所在的设备（如 CPU 或 GPU）选择对应的 backend 来执行。例如 CPU backend 会在 CPU 上运行优化过的 C++ 运算，而 GPU backend 则会调用**CUDA** 等 GPU 计算库来在显卡上执行同样的操作。

简单来说：

* **PyTorch** 是用户使用的深度学习框架和编程接口。
* **PyTorch Backend** 是在不同硬件上 **真正实现和执行这些计算操作的底层系统** 。

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=NDEzZTlkYTY1MGY4ZjNkMDc5ZTA2OTA0YTViNTVjYjZfVlF0eHhyMnc1aW5qd01JTVF6SkJqSUdJNlpYUGh4ZzZfVG9rZW46RHZjb2J5TEVKb2pWWkl4T2lzZmNuM0lVbmxiXzE3NzQxNDQ2OTY6MTc3NDE0ODI5Nl9WNA)

## 2 PyTorch 是如何管理这些 Backend的？

**PyTorch** 之所以能够同时支持 CPU、GPU 以及其他硬件，是因为它在框架内部设计了一套**统一算子 + 动态分发（dispatch）机制** 来管理不同的 Backend。整体思想是：**用户看到的是统一的算子接口，而不同硬件的实现被隐藏在 Backend 中，由框架在运行时自动选择。**

首先，PyTorch 会把所有张量操作抽象为统一的 **算子（operator）** 。例如张量加法、矩阵乘法等操作，在内部都有统一的名字（例如`aten::add`、`aten::matmul`）。这些算子只描述“要做什么计算”，而不包含具体的实现方式。换句话说，这一层只是定义计算接口，使得所有设备都可以使用同一套操作语义。

其次，不同的**Backend** 会为这些算子提供各自的实现。例如 CPU backend 会提供使用 C++ 和向量化指令优化的实现，而 GPU backend 会调用**CUDA** 等计算库在 GPU 上执行同样的算子。每个 backend 在编译或初始化时，会把自己的实现函数注册到 PyTorch 的**算子注册表** 中，例如注册“CPU 版本的 add”“CUDA 版本的 add”等。这样，框架就知道同一个算子在不同设备上有哪些可用实现。

最后，当用户在 Python 中执行一个操作时（例如`c = a + b`），PyTorch 的**Dispatcher（调度器）** 会根据参与计算的张量属性（例如设备类型 CPU 或 GPU、数据类型等）生成一个 **dispatch key** ，然后在注册表中查找最匹配的实现，并调用对应 backend 的函数。例如，如果张量位于 CPU，就调用 CPU backend 的`add` 实现；如果张量位于 GPU，就调用 CUDA backend 的实现。整个过程在运行时自动完成，因此用户不需要手动选择 backend。

因此可以把 PyTorch 的 backend 管理机制理解为三层结构： **统一算子接口 → backend 实现注册 → dispatcher 动态选择执行** 。这种设计使得 PyTorch 可以在不改变用户代码的情况下支持新的硬件平台，只需要为新的硬件实现相应的 backend 并注册到框架中即可。

这里推荐一个关于 PyTorch Dispatcher 实现机制的文章：[Let&#39;s talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDVmNTlmMjk3NjU0YWI0YjllY2Q1M2RlM2NjZjZkNGJfREk3MkplUXhxZDFEZFdRS3YwRXJoRmJ6REFHbEwxWk9fVG9rZW46UVRTcmJEdGpKb21RbGF4akRINGNDZ1l6bmdmXzE3NzQxNDQ2OTY6MTc3NDE0ODI5Nl9WNA)

## 3 想要为 PyTorch 集成一个新的 Backend 需要做些什么？

为**PyTorch** 集成一个新的 **Backend** ，本质上是让 PyTorch 能够把张量计算任务**调度到新的硬件或执行环境** 上运行。通常需要完成几个关键步骤：定义设备类型、实现算子、注册算子实现，并提供必要的运行时支持。

首先，需要在 PyTorch 中 **定义新的设备类型（device）或 dispatch key** 。PyTorch 的调度系统会根据张量所在设备（例如`cpu`、`cuda`）选择对应的 backend，因此新的 backend 需要引入一个新的 dispatch key 或 device 标识，使 PyTorch 能识别这种新的计算设备。例如，当用户创建`tensor(device="my_device")` 时，框架就知道这个张量应该由新 backend 处理。

其次，需要为该 backend**实现张量运算算子（operators）** 。PyTorch 的计算操作都是通过 ATen 算子（例如`aten::add`、`aten::matmul`）来表示的，因此新 backend 至少需要为常用算子提供对应实现。这些实现通常会调用目标硬件的计算库或驱动，例如 GPU backend 会调用**CUDA** 的计算内核。如果是新的 AI 芯片，则需要编写对应的 kernel 或 runtime 接口来完成这些计算。

第三，需要将这些算子实现**注册到PyTorch 的 dispatcher** 。每个 backend 在初始化时会把自己的算子实现注册到 PyTorch 的算子注册表中，并标明适用的 dispatch key。这样，当用户执行某个操作时，dispatcher 就可以根据张量设备自动选择新 backend 的实现并调用它。

此外，还需要实现一些 **基础运行时组件** ，例如张量内存管理（设备内存分配与释放）、数据在 CPU 和设备之间的拷贝、设备上下文管理以及流或执行队列等。这些组件保证 PyTorch 可以正确创建、存储和传输张量数据，并协调计算执行。

总体来说，为 PyTorch 集成新 backend 的核心工作包括： **定义设备类型、实现算子内核、注册到 dispatcher，以及提供设备运行时支持** 。完成这些后，用户就可以在不改变模型代码的情况下，通过指定设备来使用新的硬件 backend。很多 AI 芯片厂商实际上就是通过实现这样的 PyTorch backend 来让自己的硬件能够直接运行现有的深度学习模型。

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=Mzk1MTQxOGQyNzA3NDM1MjMxMTA0YjlhZTMzYmU1NTBfbTZQVHBCSWFEZnVhNjBZTHN4TFh1SjJCekNiejZwejlfVG9rZW46RmIwTmJ5OVNLb0NHMWl4SjVqY2M0R2UxblFlXzE3NzQxNDQ2OTY6MTc3NDE0ODI5Nl9WNA)

上图中的 PyTorch 官方文档主要讲解了**如何在 PyTorch 中集成新的硬件加速器**  **（Backend）** ，内容覆盖从低层到高层的完整流程：包括 **运行时组件** （Event、Stream、Memory、Generator、Guard、Hooks 等 C++ 支撑结构）、 **算子实现** （最小算子集、前向/后向、fallback、STUB 等）、 **Python 前端** （设备无关的 API 和绑定）、以及 **高层模块集成** （如 AMP、Compiler、ONNX、分布式等），并通过官方参考实现和示例说明如何注册 backend、管理设备和调度算子，让开发者可以在不修改上游代码的情况下接入自己的加速器。

## 4 我想自己试着为 PyTorch 集成 Backend 并自定义算子该怎么办？

### 4-1 没有真实硬件加速器没关系！

如果没有真实硬件加速器，也想体验为 PyTorch 集成新 Backend，可以用 **OpenReg** ：它是一个在 CPU 上模拟 CUDA 行为的后端库，提供和 CUDA Runtime 一致的 API（设备管理、内存管理、流和事件），用多线程和内存保护模拟 GPU 的设备上下文和任务调度。这样，我们就可以在 CPU 上原型化、调试和测试 PyTorch 的 Backend 集成流程（包括算子注册、Dispatcher 调度和 PrivateUse1 后端），而无需真实硬件，只需编译 OpenReg 库并运行示例程序就能验证整个流程。

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=MmNiMDExZTA5ZTMzMjZkMzg0YWFhYjZhZTk0NjQ1YWZfUUJlN212VVNkRVg4cjI1VlhMT09TYjBMY2hIdVUxUHJfVG9rZW46VmRkemI2MjBsb2NNMEh4VnVBUmMwYXdPblBmXzE3NzQxNDQ2OTY6MTc3NDE0ODI5Nl9WNA)

### 4-2 具体怎么做？

没有真实硬件时，可以先用 OpenReg 跑通一遍 PyTorch 自定义 Backend 的基本流程。我的实验步骤如下。

首先启动一个 Ubuntu 22.04 的 Docker 环境：

```SQL
docker pull ubuntu:22.04

docker run -it \
--rm \
--privileged \
--network=host \
--ipc=host \
--shm-size=16G \
-v ~/docker-file:/workspace \
--cap-add=SYS_ADMIN \
--security-opt seccomp=unconfined \
ubuntu:22.04 \
/bin/bash -c "exec /bin/bash"
```

进入容器后，安装基础依赖：

```Python
apt update
apt upgrade -y
apt install python3 python3-pip vim cmake git -y
```

然后安装 PyTorch CPU 版本：

```Python
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

先确认 PyTorch 本身可以正常运行：

```Python
import torch
x = torch.rand(5, 3)
print(x)
```

接着获取 PyTorch 代码，并进入 OpenReg 对应的扩展目录进行安装：

```Python
git clone https://github.com/pytorch/pytorch.git
cd pytorch
cd test/cpp_extensions/open_registration_extension/torch_openreg
python3 -m pip install --no-build-isolation .
```

安装完成后，可以用下面的脚本测试 OpenReg backend 是否可用：

```Python
import torch

if not hasattr(torch, "openreg") or not torch.openreg.is_available():
    print("OpenReg backend is not available.")
    exit()
print("OpenReg backend is available!")

device = torch.device("openreg")

x = torch.tensor([[1., 2.], [3., 4.]], device=device)
y = x + 2
print("Result y:\n", y)
print(f"Device of y: {y.device}")

z = y.cpu()
print("Result z:\n", z)
print(f"Device of z: {z.device}")
```

如果这段代码能够正常运行，就说明这条最小链路已经打通了：

* openreg 设备已被 PyTorch 识别；
* x + 2 会通过 Dispatcher 分发到 OpenReg backend；
* 结果可以再通过 .cpu() 拷回 CPU 进行验证。

虽然这个例子很小，但已经覆盖了自定义 Backend 的几个关键点：设备注册、算子调度和数据搬运。

如果想要为其注册新的算子或实现更多高级功能，直接对代码进行相应修改即可。

## 5 总结

为 PyTorch 集成自定义 Backend，核心是把新的设备类型、算子实现和运行时支持接入 PyTorch 的 dispatcher 体系中。这样用户仍然使用统一的 PyTorch 接口，而具体执行则由框架自动分发到对应 backend。

OpenReg 的意义在于：即使没有真实硬件，也可以先把这条链路完整跑通。通过这个最小实验，可以更直观地理解 PyTorch Backend 是如何被注册、调度和使用的。这也是进一步实现真实硬件 backend 的一个很好起点。
