# gdb

gdb的功能很强大，这里介绍一个：当执行二进制程序时，出现段错误，怎么将发生段错误的位置定位到源码？

## 1 重新源码编译构建

在编译构建时，需要加上-g调试选项。

## 2 生成core文件

需要修改一些配置，让系统在运行程序时可以生成core文件：

```shelll
ulimit -c unlimited
echo "* soft core unlimited" >> /etc/security/limits.conf
```

配置好之后，运行程序，出发段错误。

## 3 gdb分析core

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
