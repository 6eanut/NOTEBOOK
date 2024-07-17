#!/bin/bash

# 获取当前系统架构
ARCH=$(uname -m)

# 判断架构类型
if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ]]; then
    echo "当前系统是 ARM 架构"
elif [[ "$ARCH" == "riscv64" ]]; then
    echo "当前系统是 RISC-V 架构"
else
    echo "当前系统不是 ARM 或 RISC-V 架构，而是 $ARCH"
    exit 1
fi

mkdir /usr/local/python-3.11.6 -p
wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tar.xz
tar -xJvf Python-3.11.6.tar.xz
cd Python-3.11.6

# 判断架构类型
if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ]]; then
    ./configure --prefix=/usr/local/python-3.11.6 CFLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
elif [[ "$ARCH" == "riscv64" ]]; then
    ./configure --prefix=/usr/local/python-3.11.6 CFLAGS="-fno-omit-frame-pointer"
else
    echo "当前系统不是 ARM 或 RISC-V 架构，而是 $ARCH"
    exit 1
fi

make -j $(nproc)
make install -j $(nproc)