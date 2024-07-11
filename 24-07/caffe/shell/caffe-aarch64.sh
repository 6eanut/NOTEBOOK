#!/bin/sh

set -e # 在遇到非零返回值时立即退出

echo "step1: 安装依赖包"
# 如果当前用户没有sudo权限，则注释掉下面两行，改用root用户来运行下面两行
sudo dnf install -y leveldb-devel snappy-devel opencv.aarch64 boost-devel hdf5-devel gflags-devel glog-devel lmdb-devel openblas.aarch64
sudo dnf install -y git wget tar gcc-c++ unzip automake libtool autoconf

echo "build protobuf-3.9.x from source"
mkdir -p protobuf39x && cd protobuf39x
mkdir install
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout 3.9.x
./autogen.sh
WORKSPACE=$(pwd)
./configure --prefix=$WORKSPACE/../install
make -j $(nproc)
make install

PROTOBUF="$WORKSPACE/../install"
PATH="$PROTOBUF/bin:$PATH"
LD_LIBRARY_PATH="$PROTOBUF/lib:$LD_LIBRARY_PATH"

echo "Adding environment variables to ~/.bashrc"
echo "# protobuf" >> ~/.bashrc
echo "export PROTOBUF=$PROTOBUF" >> ~/.bashrc
echo "export PATH=$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
protoc --version || { echo "Protobuf build failed"; exit 1; }

cd ../..

echo "step2: 编译caffe"
git clone https://github.com/BVLC/caffe.git
cd caffe
wget https://raw.githubusercontent.com/6eanut/NOTEBOOK/main/24-07/caffe/makefiles/aarch64-Makefile.config
mv aarch64-Makefile.config Makefile.config
# 适配opencv4.x
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/layers/window_data_layer.cpp
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/util/io.cpp
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::ImreadModes::IMREAD_GRAYSCALE/g' src/caffe/util/io.cpp
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/test/test_io.cpp
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::ImreadModes::IMREAD_GRAYSCALE/g' src/caffe/test/test_io.cpp

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
