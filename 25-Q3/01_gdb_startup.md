# gdb

## 1 段错误

gdb的功能很强大，这里介绍一个：当执行二进制程序时，出现段错误，怎么将发生段错误的位置定位到源码？

### 1-1 重新源码编译构建

在编译构建时，需要加上-g调试选项。

### 1-2 生成core文件

需要修改一些配置，让系统在运行程序时可以生成core文件：

```shelll
ulimit -c unlimited
echo "* soft core unlimited" >> /etc/security/limits.conf
```

配置好之后，运行程序，触发段错误。

### 1-3 gdb分析core

```shell
# 查看core文件
coredumpctl list

# 导出core文件
coredumpctl dump /usr/bin/file -o ./file.core

# 使用gdb调试
gdb /usr/bin/file ./file.core

# 查看调用栈
where
```

## 2 QEMU

### 2-1 启动qemu失败

当执行一条qemu命令时，可能会启动失败，这时可用gdb来调试查看调用栈。

```shell
# 获取进程号
pgrep -af qemu-system-x86_64
# gdb调试
sudo gdb -p pid
# qemu是多线程启动，获取所有现成的backtrace
thread apply all bt

# 或者直接在gdb中运行qemu
sudo gdb --args qemu-system-x86_64 ...
```

### 2-2 在qemu中执行程序失败

在启动qemu时加上-s -S选项，表示在1234端口启动GDB服务器且启动时暂停CPU等待GDB连接

```
qemu-system-riscv64 \
    -machine virt \
    -kernel your_kernel_image \
    -append "console=ttyS0 root=/dev/vda ro" \
    -nographic \
    -s -S
```

在host端：

```shell
gdb-multiarch vmlinux		# 进入gdb界面
(gdb) target remote :1234	# 连接服务器
(gdb) continue 			# qemu继续启动
(gdb) break walk_stackframe	# 设置断点，如果已知内核崩溃发生在某个特定函数
(gdb) continue

(gdb) bt			# 查看调用栈
(gdb) list			# 查看当前执行的代码

(gdb) i b			# 查看设置的断点信息
(gdb) d 1			# 删除断点

(gdb) step			# 执行下一行代码，会跳进函数内部
(gdb) next			# 执行下一行代码，不会跳进函数内部
(gdb)fin			# 执行完当前函数，并停在函数返回后的位置
```
