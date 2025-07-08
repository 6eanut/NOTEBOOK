# 复现

## docker容器创建

```
docker pull ubuntu:20.04
docker run -v /home/syzdirect/svfdirect_container:/home -it --device=/dev/kvm ubuntu:20.04 /bin/bash
apt update
apt install -y sudo cmake make gcc g++ wget vim flex bison libncurses-dev libelf-dev libssl-dev git bc qemu-system-x86 libboost-all-dev git build-essential python3-dev python-is-python3 python3-pip clang llvm lldb lld bc
pip install pandas Levenshtein tqdm openpyxl

export PATH=/home/SyzDirect/source/llvm-project-new/build/bin:$PATH
export LD_LIBRARY_PATH=/home/SyzDirect/source/llvm-project-new/build/lib:$LD_LIBRARY_PATH
export LLVM_SYMBOLIZER_PATH=/home/SyzDirect/source/llvm-project-new/build/bin/llvm-symbolizer
export CPATH=/home/SyzDirect/source/llvm-project-new/llvm/include:$CPATH

cd /home
wget https://dl.google.com/go/go1.22.1.linux-amd64.tar.gz
tar -xf go1.22.1.linux-amd64.tar.gz
export GOROOT=/home/go
export GOPATH=/home/go
export PATH=$GOROOT/bin:$PATH
export GOPROXY=https://goproxy.cn,direct
```

## 复现步骤

```shell
cd /home
git clone git@github.com:seclab-fudan/SyzDirect.git

cd SyzDirect/source/syzdirect/Runner
mkdir workdir
cd workdir
mkdir srcs
cd srcs
git clone git@github.com:j1akai/case_0.git

# 在host下创好images，然后传送给docker，参考https://github.com/j1akai/ConfigFuzz/tree/jiakai-dev/SyzDirect/source#prepare-fileimage
sudo chown -R root:root /home/images/ 
sudo chmod 700 /home/images/ 
sudo chmod 600 /home/images/bullseye.id_rsa

# 修改SyzDirect/source/syzdirect/Runner/Config.py的内容
# CleanImageTemplatePath="/home/images/bullseye.img"
# KeyPath="/home/images/bullseye.id_rsa"

# 修改SyzDirect/source/syzdirect/syzdirect_fuzzer/sys/sys.go

cd /home/SyzDirect/source/syzdirect/Runner
python3 Main.py prepare_kernel_bitcode -j 8 > /home/log/pkb.log 2>&1
python3 Main.py analyze_kernel_syscall -j 8 > /home/log/aks.log 2>&1
python3 Main.py extract_syscall_entry -j 8 > /home/log/ese.log 2>&1
python3 Main.py instrument_kernel_with_distance -j 8 > /home/log/ikwd.log 2>&1
python3 Main.py fuzz -j 8 > /home/log/fuzz.log 2>&1
```
