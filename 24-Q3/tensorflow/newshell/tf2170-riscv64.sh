/usr/local/python-3.11.6/bin/python3.11 -m venv venv00
source venv00/bin/activate
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps

wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-dist.zip
mkdir bazel-dist; cd bazel-dist
mv ../bazel-6.5.0-dist.zip .
unzip bazel-6.5.0-dist.zip
rm bazel-6.5.0-dist.zip
wget bazel-6.5.0-dist-riscv-v0.patch
patch -p1 < bazel-6.5.0-dist-riscv-v0.patch
EMBED_LABEL="6.5.0" EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" ./compile.sh

git clone https://gitee.com/xu-jia_kai/tensorflow.git
cd tensorflow
git checkout tags/v2.17.0
./configure
export TF_PYTHON_VERSION=3.11
bazel build //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu --local_ram_resources=800 --jobs=4
