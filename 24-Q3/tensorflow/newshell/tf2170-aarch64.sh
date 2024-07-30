/usr/local/python-3.11.6/bin/python3.11 -m venv venv00
source venv00/bin/activate
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps

# only aarch64
mkdir install; cd install
wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-arm64
mv bazel-6.5.0-linux-arm64 bazel
chmod +x bazel
export PATH=$(pwd):$PATH
cd ..

# riscv: git clone https://gitee.com/xu-jia_kai/tensorflow.git
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v2.17.0
./configure
export TF_PYTHON_VERSION=3.11
bazel build //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow --local_ram_resources=600 --jobs=4
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.17.0-cp311-cp311-linux_aarch64.whl

git clone https://github.com/tensorflow/text.git
cd text
git checkout tags/v2.17.0
