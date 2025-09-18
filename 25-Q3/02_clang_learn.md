# Clang

需求是这样的：在X86的机器上，用Clang从源码编译构建RISCV64的Linux。

## 第一步：能不能在X86的机器上，用Clang从源码编译一个C语言程序？

首先准备一个C语言程序：

```c
#include <stdio.h>
int main() {
    printf("Hello, RISC-V!\n");
    return 0;
}
```

然后从https://github.com/riscv-collab/riscv-gnu-toolchain/releases里面下载RISC-V的sysroot

然后运行以下命令即可编译：

```shell
clang \
    --target=riscv64-unknown-linux-gnu \
    --sysroot=/home/temp/tools/riscv/sysroot \
    -B/home/temp/tools/riscv/lib/gcc/riscv64-unknown-linux-gnu/12.2.0 \
    -L/home/temp/tools/riscv/lib/gcc/riscv64-unknown-linux-gnu/12.2.0 \
    -o hello_riscv \
    hello.c
```

## 第二步：在X86上，用Clang从源码编译构建Linux

```shell
make LLVM=1 ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- defconfig
make LLVM=1 ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu-
```

可以查看.config文件来确定用到的clang版本：

```config
#
# Automatically generated file; DO NOT EDIT.
# Linux/riscv 6.6.0 Kernel Configuration
#
CONFIG_CC_VERSION_TEXT="clang version 13.0.1"
CONFIG_GCC_VERSION=0
CONFIG_CC_IS_CLANG=y
CONFIG_CLANG_VERSION=130001
CONFIG_AS_IS_LLVM=y
CONFIG_AS_VERSION=130001
CONFIG_LD_VERSION=0
CONFIG_LD_IS_LLD=y
CONFIG_LLD_VERSION=130001
CONFIG_CC_HAS_ASM_GOTO_OUTPUT=y
CONFIG_TOOLS_SUPPORT_RELR=y
CONFIG_CC_HAS_ASM_INLINE=y
CONFIG_CC_HAS_NO_PROFILE_FN_ATTR=y
CONFIG_PAHOLE_VERSION=121
CONFIG_IRQ_WORK=y
CONFIG_BUILDTIME_TABLE_SORT=y
CONFIG_THREAD_INFO_IN_TASK=y
```

可以看到是riscv架构，且版本是本地clang的13.0.1版本。
