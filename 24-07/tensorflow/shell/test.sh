/usr/local/python-3.11.6/bin/python3.11 -m venv venv00
source venv00/bin/activate

# 安装 TensorFlow pip 软件包依赖项
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps

git clone https://gitee.com/xu-jia_kai/src-openEuler-bazel.git
cd src-openEuler-bazel
git checkout openEuler-23.09
wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-dist.zip
# patch文件参考 https://gitee.com/src-openeuler/bazel/pulls/29/files
vi 04-riscv-distdir_deps.patch
vi abseil-cpp-riscv.patch
# 根据自身情况修改
vi bazel.spec

dnf install rpm-build.aarch64
dnf install bash-completion.noarch
mkdir -p /home/tf2150-bfs/rpmbuild/SOURCES/
cp bazel-6.1.0-dist.zip 0* linux-bazel-path-from-getauxval.patch abseil-cpp-riscv.patch /home/tf2150-bfs/rpmbuild/SOURCES/
rpmbuild -ba bazel.spec
sudo rpm -ivh /home/tf2150-bfs/rpmbuild/RPMS/aarch64/bazel-6.1.0-2.aarch64.rpm

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v2.15.0
./configure
export TF_PYTHON_VERSION=3.11
bazel build //tensorflow/tools/pip_package:build_pip_package --local_ram_resources=1024 --jobs=4

# rpm -qa | grep bazel 查看bazel的rpm包