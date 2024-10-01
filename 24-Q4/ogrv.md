# OpenGauss RISC-V

> 本文档主要用来参考搭建OpenGauss RISC-V

演示环境：[Fedora39](https://hub.docker.com/r/imbearchild/fedora-rv64)（Docker环境）

## 0-环境准备

### Ⅰ-基础软件包

```
dnf update -y
dnf install wget tar gzip gcc gcc-g++ lbzip2 bison flex texinfo expect diffutils -y
```

### Ⅱ-gcc10.3

```
mkdir /user/local/gcc-10.3.0 -p
cd /opt
wget https://mirrors.aliyun.com/gnu/gcc/gcc-10.3.0/gcc-10.3.0.tar.gz
tar -xvf gcc-10.3.0.tar.gz
cd gcc-10.3.0/
./contrib/download_prerequisites
# 参考https://github.com/gcc-mirror/gcc/commit/2701442d0cf6292f6624443c15813d6d1a3562fe做修改
mkdir build; cd build
../configure --prefix=/user/local/gcc-10.3.0 --enable-shared --build=riscv64-unknown-linux-gnu --target=riscv64-unknown-linux-gnu --enable-languages=c,c++ --disable-multilib
make; make install
```

### Ⅲ-python3.11.6
