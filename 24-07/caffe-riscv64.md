# Caffe on RISC-V

[参考指南](caffe-aarch64.md)

## 1-环境搭建

### step1：安装依赖包

```
sudo dnf install -y leveldb-devel snappy-devel opencv.riscv64 boost-devel hdf5-devel gflags-devel glog-devel lmdb-devel openblas.riscv64 protobuf-devel.riscv64
sudo dnf install -y git wget tar gcc-c++ unzip automake libtool autocon
```

在riscv平台上，采用dnf install的方式对protobuf进行了安装，原因是可以通过修改caffe的源码来完成caffe对高版本protobuf的适配。

### step2：编译caffe

```
git clone https://github.com/BVLC/caffe.git
cd caffe
# 这一步用来解决高版本protobuf和caffe不适配的问题
sed -i 's/coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);/coded_input->SetTotalBytesLimit(kProtoReadBytesLimit);/g' src/caffe/util/io.cpp
# 下载适配于riscv64的Makefile.config
wget https://raw.githubusercontent.com/6eanut/NOTEBOOK/main/24-07/caffe/makefiles/riscv64-Makefile.config
mv riscv64-Makefile.config Makefile.config
# 修改caffe源码以适配opencv4.x
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/layers/window_data_layer.cpp
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/util/io.cpp
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::ImreadModes::IMREAD_GRAYSCALE/g' src/caffe/util/io.cpp
sed -i 's/CV_LOAD_IMAGE_COLOR/cv::IMREAD_COLOR/g' src/caffe/test/test_io.cpp
sed -i 's/CV_LOAD_IMAGE_GRAYSCALE/cv::ImreadModes::IMREAD_GRAYSCALE/g' src/caffe/test/test_io.cpp
# 设置LDFLAGS环境变量，否则在make test时会出现链接器未能正确地找到libgfortran.so.5库的报错
export LDFLAGS="-L/usr/lib -Wl,-rpath=/usr/lib -lgfortran"
make all -j $(nproc)
make test -j $(nproc)
make runtest -j $(nproc)
```

在aarch64上，采用降低protobuf版本的方式解决了caffe和protobuf版本不适配的问题；在riscv64上，采用修改caffe源码的方式解决了两者不适配的问题。

> 环境搭建自动化[脚本](caffe/shell/caffe-riscv64.sh)

## 2-用例运行

下载deploy文件->下载perf测试脚本->对deploy文件分别进行time

> 用例运行自动化[脚本](caffe/shell/caffe-test.sh)
