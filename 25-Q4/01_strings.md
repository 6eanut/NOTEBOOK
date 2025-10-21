# strings

strings是一个实用的小工具，它用来从二进制文件中提取出可打印的文本字符串。

## 1 二进制文件

二进制文件(比如可执行文件、内核镜像)里除了机器码和数据之外，可能还包含一些人类可读的文本，比如版本号、配置信息等。strings会扫描整个文件，把连续的ASCII或UTF-8可打印字符提取出来显示。

## 2 使用方法

```shell
strings <二进制文件>
```

对于内核镜像而言：

```shell
strings vmlinux | grep "Linux version"
```
