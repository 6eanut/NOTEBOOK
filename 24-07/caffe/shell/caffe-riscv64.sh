#!/bin/sh

set -e # 在遇到非零返回值时立即退出

echo "step1: 安装依赖包"
sudo dnf install -y leveldb-devel snappy-devel opencv.riscv64 boost-devel hdf5-devel gflags-devel glog-devel lmdb-devel openblas.riscv64
sudo dnf install -y git wget tar gcc-c++ unzip automake libtool autoconf

echo "step2: 编译caffe"
git clone https://github.com/BVLC/caffe.git
cd caffe
wget https://raw.githubusercontent.com/6eanut/caffe-makefile/main/Makefile.config
# 适配opencv4.x
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/layers/window_data_layer.cpp
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/util/io.cpp
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::ImreadModes::IMREAD_GRAYSCALE/g' src/caffe/util/io.cpp
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/test/test_io.cpp
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::ImreadModes::IMREAD_GRAYSCALE/g' src/caffe/test/test_io.cpp

export LDFLAGS="-L/usr/lib -Wl,-rpath=/usr/lib -lgfortran"
make all -j $(nproc)
make test -j $(nproc)
make runtest -j $(nproc)

CAFFE=$(pwd)
PATH="$CAFFE/build/tools:$PATH"

echo "Adding environment variables to ~/.bashrc"
echo "# caffe" >> ~/.bashrc
echo "export CAFFE=$CAFFE" >> ~/.bashrc
echo "export PATH=$PATH" >> ~/.bashrc
source ~/.bashrc
caffe --version || { echo "Caffe build failed"; exit 1; }