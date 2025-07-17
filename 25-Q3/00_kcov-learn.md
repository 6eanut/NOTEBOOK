# KCOV: code coverage for fuzzing

原文：[https://github.com/torvalds/linux/blob/master/Documentation/dev-tools/kcov.rst](https://github.com/torvalds/linux/blob/master/Documentation/dev-tools/kcov.rst)

## 1 概述

KCOV是Linux内核的一个配置项，在打开该配置项的Linux内核上，可以捕获执行某个系统调用时，该系统调用对于内核代码路径的覆盖范围，捕获到的信息放在kcov这个debugfs虚拟文件中。

KCOV的正确执行依赖于在编译Linux内核时，编译器对程序的插桩，这要求GCC版本是在6.1.0之后(或任何Clang版本)。

在编译Linux内核时，需要打开CONFIG_KCOV这个配置项。

进入操作系统后，需要挂在debugfs这个虚拟文件，否则无法访问捕获到的覆盖信息：

```shell
mount -t debugfs none /sys/kernel/debug
```

## 2 例子

下面通过一个具体实例，说明如何利用KCOV来收集一个系统调用的覆盖范围。

```c
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/types.h>

#define KCOV_INIT_TRACE                     _IOR('c', 1, unsigned long)
#define KCOV_ENABLE                 _IO('c', 100)
#define KCOV_DISABLE                        _IO('c', 101)
#define COVER_SIZE                  (64<<10)

#define KCOV_TRACE_PC  0
#define KCOV_TRACE_CMP 1

int main(int argc, char **argv)
{
    int fd;
    unsigned long *cover, n, i;

    /* A single fd descriptor allows coverage collection on a single
     * thread.
     */
    fd = open("/sys/kernel/debug/kcov", O_RDWR);
    if (fd == -1)
            perror("open"), exit(1);
    /* Setup trace mode and trace size. */
    if (ioctl(fd, KCOV_INIT_TRACE, COVER_SIZE))
            perror("ioctl"), exit(1);
    /* Mmap buffer shared between kernel- and user-space. */
    cover = (unsigned long*)mmap(NULL, COVER_SIZE * sizeof(unsigned long),
                                 PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if ((void*)cover == MAP_FAILED)
            perror("mmap"), exit(1);
    /* Enable coverage collection on the current thread. */
    if (ioctl(fd, KCOV_ENABLE, KCOV_TRACE_PC))
            perror("ioctl"), exit(1);
    /* Reset coverage from the tail of the ioctl() call. */
    __atomic_store_n(&cover[0], 0, __ATOMIC_RELAXED);
    /* Call the target syscall call. */
    read(-1, NULL, 0);
    /* Read number of PCs collected. */
    n = __atomic_load_n(&cover[0], __ATOMIC_RELAXED);
    for (i = 0; i < n; i++)
            printf("0x%lx\n", cover[i + 1]);
    /* Disable coverage collection for the current thread. After this call
     * coverage can be enabled for a different thread.
     */
    if (ioctl(fd, KCOV_DISABLE, 0))
            perror("ioctl"), exit(1);
    /* Free resources. */
    if (munmap(cover, COVER_SIZE * sizeof(unsigned long)))
            perror("munmap"), exit(1);
    if (close(fd))
            perror("close"), exit(1);
    return 0;
}
```

上面的程序主要做的事情有：

* 打开KCOV设备：`fd = open("/sys/kernel/debug/kcov", O_RDWR);`
* 初始化跟踪缓冲区：`ioctl(fd, KCOV_INIT_TRACE, COVER_SIZE);`
* 内存映射共享缓冲区：`cover = mmap(..., MAP_SHARED, fd, 0);`
* 启用覆盖率收集：`ioctl(fd, KCOV_ENABLE, KCOV_TRACE_PC);`
* 触发目标系统调用：`read(-1, NULL, 0); // 故意调用一个无效的 read()`
* 读取覆盖率数据：`n = __atomic_load_n(&cover[0], __ATOMIC_RELAXED);`
  `for (i = 0; i < n; i++) printf("0x%lx\n", cover[i + 1]);`
* 清理资源

在guest上执行该程序，会得到如下打印信息：

```shell
0xffffffff81aad351
0xffffffff81aad19a
0xffffffff81b2e849
0xffffffff81b2e8c5
0xffffffff81b2e9ec
0xffffffff81aad2a3
0xffffffff813bff7d
0xffffffff813bffb6
0xffffffff813bffe6
0xffffffff813bffd8
```

在host上执行如下命令，解析上述信息到文件和行号：

```shell
$ addr2line -e /home/syzdirect/linux/vmlinux \
  0xffffffff81aad351 \
  0xffffffff81aad19a \
  0xffffffff81b2e849 \
  0xffffffff81b2e8c5 \
  0xffffffff81b2e9ec \
  0xffffffff81aad2a3 \
  0xffffffff813bff7d \
  0xffffffff813bffb6 \
  0xffffffff813bffe6 \
  0xffffffff813bffd8
/home/syzdirect/linux/fs/read_write.c:720
/home/syzdirect/linux/./include/linux/file.h:85
/home/syzdirect/linux/./arch/x86/include/asm/current.h:23
/home/syzdirect/linux/./include/linux/fdtable.h:74 (discriminator 2)
/home/syzdirect/linux/fs/file.c:1216
/home/syzdirect/linux/fs/read_write.c:703
/home/syzdirect/linux/./arch/x86/include/asm/current.h:23
/home/syzdirect/linux/arch/x86/kernel/fpu/context.h:38 (discriminator 5)
/home/syzdirect/linux/arch/x86/kernel/fpu/context.h:38 (discriminator 11)
/home/syzdirect/linux/arch/x86/kernel/fpu/core.c:832
```

然后喂给LLM，它给我分析了一下这个路径：

> **系统调用入口**：`read_write.c`处理系统调用参数
>
> **文件描述符处理**：通过 `file.h`和 `fdtable.h`获取文件对象。
>
> **进程上下文管理**：`current.h`获取当前进程信息。
>
> **文件操作**：`file.c`处理文件引用和描述符表。
>
> **FPU 状态管理**：在系统调用返回前恢复 FPU 上下文。

## 3 心得

Syzkaller在执行种子时，也是需要打开KCOV设备、初始化跟踪缓冲区、内存映射共享缓冲区、启用覆盖率收集，然后才去执行种子，最后再去读覆盖率数据。这是一个基本的执行流程。

/sys/kernel/debug/kcov是一个接口文件，只能通过特定的ioctl操作与用户程序交互，不支持直接读写文件内容(cat echo等)。所以想要读取覆盖信息，必须通过open打开文件，通过ioctl和mmap操作收集覆盖率。
