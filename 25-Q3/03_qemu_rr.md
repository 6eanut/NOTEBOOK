# Record and Replay in QEMU

虚拟机的Record和Replay是一种记录和回放的技术，它能够将虚拟机在某个时间段内的所有输入和状态变化完整地记录下来，这样开发者就可以随时反复地将虚拟机精确地回放到记录期间的任何一个时间点状态。

这对于故障复现和调试很有意义，即一个bug在测试环境中偶尔发生，难以稳定复现，开发者无法调试；测试人员开启replay记录功能进行测试，一旦故障发生，就可以将完整的记录文件交给开发人员，开发者可以在自己的机器上回放故障发生前的整个过程，从而定位问题根源。

QEMU作为开源的虚拟化解决方案，提供了record和replay的命令行选项，供开发者使用。

## 第一阶段：record

记录阶段的目标是创建一个录制文件，该文件用于第二阶段的回放

```shell
qemu-system-x86_64 \
    -m 2G \
    -kernel pathto/linux/arch/x86/boot/bzImage \
    -append "console=ttyS0 root=/dev/sda earlyprintk=serial net.ifnames=0" \
    -drive file=pathto/images/bullseye.img,format=raw \
    -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:10025-:22 \
    -net nic,model=e1000 \
    -nographic \
    -pidfile vm.pid \
    -icount shift=auto,rr=record,rrfile=my_recording.bin \
    -monitor none \
    2>&1 | tee vm_record.log
```

其中-icount shift=auto,rr=record,rrfile=my_recording.bin，表示为记录模式，录制文件保存为my_recording.bin

## 第二阶段：replay

首先使用下面的回放模式命令启动qemu

```shell
qemu-system-x86_64 \
    -m 2G \
    -kernel pathto/linux/arch/x86/boot/bzImage \
    -append "console=ttyS0 root=/dev/sda earlyprintk=serial net.ifnames=0" \
    -drive file=pathto/images/bullseye.img,format=raw \
    -net user,host=10.0.2.10,hostfwd=tcp:127.0.0.1:10025-:22 \
    -net nic,model=e1000 \
    -nographic \
    -pidfile vm.pid \
    -icount shift=auto,rr=replay,rrfile=my_recording.bin \
    -S -s \
    2>&1 | tee vm_replay.log
```

rr=replay表示为回放模式，-S -s表示启动时暂停，等待调试器命令(在localhost:1234开启GDB调试服务)。

然后打开另一个终端，使用GDB连接并调试

```shell
# 切换到你的内核源码编译目录
cd pathto/linux

# 启动 GDB 并加载内核符号文件
gdb ./vmlinux

# 在 GDB 中连接到 QEMU
(gdb) target remote localhost:1234

# 设置断点（例如在 panic 函数处）
(gdb) hb panic

# 开始执行
(gdb) continue
```

VM现在会开始回放，直到触发panic函数时暂停，此时可以在GDB中检查调用栈、变量、寄存器等信息。

使用record和replay有以下问题：

* kvm和record相矛盾，所以启用record时，kvm必须关闭，这会导致运行速度变慢；
* 录制文件大小随着时间越长，文件越大；
