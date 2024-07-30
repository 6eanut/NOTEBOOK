# 0-pythin虚拟环境的创建
/usr/local/python-3.11.6/bin/python3.11 -m venv venv00
source venv00/bin/activate
# 安装 TensorFlow pip 软件包依赖项
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps

# 1-构建bazel
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

# 2.0-构建tensorflow 官方
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout tags/v2.15.0
./configure
export TF_PYTHON_VERSION=3.11
bazel build //tensorflow/tools/pip_package:build_pip_package --local_ram_resources=1024 --jobs=4

# rpm -qa | grep bazel 查看bazel的rpm包
# rpm -e bazel-6.1.0-2.riscv64 产出rpm包
# rpm -ivh /home/test00/rpmbuild530/RPMS/riscv64/bazel-5.3.0-2.riscv64.rpm 装新的包

# 2.1-构建tensorflow 非官方
git clone https://gitee.com/src-openeuler/tensorflow.git
cd tensorflow
git checkout openEuler-24.03-LTS


# apt install perl
# curl -L -o abseil-cpp-20230802.0.tar.gz https://gitee.com/xu-jia_kai/abseil-cpp/archive/refs/tags/20230802.0.tar.gz
# shasum -a 256 abseil-cpp-20230802.0.tar.gz