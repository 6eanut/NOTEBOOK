#!/bin/sh

set -e # 在遇到非零返回值时立即退出

start=$(date +%s)

# 安装 Python 和 TensorFlow 软件包依赖项
sudo dnf install python3-devel python3-pip -y
# 安装 bazel 软件包依赖项
sudo dnf install python zip unzip java-11-openjdk-devel -y
# 安装基本的工具
sudo dnf install patchelf openssl-devel -y

# 创建虚拟环境
cd /opt
sudo ./python_3.11.6_build.sh
cd ~
/usr/local/python-3.11.6/bin/python3.11 -m venv venv00
python -m venv venv00
source venv00/bin/activate

# 安装 TensorFlow pip 软件包依赖项
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps

# 安装 bazel 6.1.0 , TensorFlow 需要
mkdir bazel-source; cd bazel-source
wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-dist.zip
unzip bazel-6.1.0-dist.zip
rm bazel-6.1.0-dist.zip
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.22.7-2.oe2403.riscv64
./compile.sh
./output/bazel

BAZEL=$(pwd)/output
PATH="$BAZEL:$PATH"
echo "# bazel" >> ~/.bashrc
echo "export BAZEL=$BAZEL" >> ~/.bashrc
echo "export PATH=$PATH" >> ~/.bashrc
source ~/.bashrc

cd ..

# 下载 TensorFlow 源码
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v2.15.0
./configure
# 创建软链接
sudo ln -s /usr/bin/aarch64-openEuler-linux-gcc /usr/bin/aarch64-linux-gnu-gcc
export TF_PYTHON_VERSION=3.11
bazel build //tensorflow/tools/pip_package:build_pip_package --local_ram_resources=1024 --jobs=12
WORKSPACE=$(pwd)
./bazel-bin/tensorflow/tools/pip_package/build_pip_package $WORKSPACE/../tensorflow_pkg
pip install $WORKSPACE/../tensorflow_pkg/tensorflow*

end=$(date +%s)
runtime=$((end-start))
echo "脚本执行时长： $runtime s"