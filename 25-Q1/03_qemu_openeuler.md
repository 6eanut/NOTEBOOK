# 搭建QEMU OpenEuler环境

## 1 RISC-V

### 1-1 下载QEMU源码

```
cd ~
mkdir cache
cd ~/cache
wget -c https://download.qemu.org/qemu-9.0.0.tar.xz
tar -xvf qemu-9.0.0.tar.xz
```

### 1-2 编译安装QEMU

```
mkdir ~/path
mkdir ~/cache/qemu-9.0.0-build
cd ~/cache/qemu-9.0.0-build
../qemu-9.0.0/configure --target-list=riscv64-softmmu,riscv64-linux-user --prefix=/home/$(whoami)/path
make -j $(nproc)
make install
export PATH="/home/$(whoami)/path/bin:$PATH"
```

### 1-3 下载OERV

```
mkdir ~/oerv_qemu
cd ~/oerv_qemu
wget https://repo.openeuler.org/openEuler-24.03-LTS/virtual_machine_img/riscv64/RISCV_VIRT_CODE.fd
wget https://repo.openeuler.org/openEuler-24.03-LTS/virtual_machine_img/riscv64/RISCV_VIRT_VARS.fd
wget https://repo.openeuler.org/openEuler-24.03-LTS/virtual_machine_img/riscv64/fw_dynamic_oe_2403_penglai.bin
wget https://repo.openeuler.org/openEuler-24.03-LTS/virtual_machine_img/riscv64/openEuler-24.03-LTS-riscv64.qcow2.xz
xz -d openEuler-24.03-LTS-riscv64.qcow2.xz
# 下载完成后，可以按需修改start_vm.sh文件中的参数
wget https://repo.openeuler.org/openEuler-24.03-LTS/virtual_machine_img/riscv64/start_vm.sh
chmod +x start_vm.sh
```

### 1-4 启动OERV_QEMU

```
./start_vm.sh
# root用户密码默认为openEuler12#$
ssh -p 12055 root@localhost
```

## 2 Aarch64

### 2-1 安装QEMU

```shell
sudo apt update
sudo apt install qemu-system-arm qemu-utils
```

### 2-2 下载OpenEuler Aarch64

```shell
mkdir ~/oeaarch64_qemu
cd ~/oeaarch64_qemu
wget https://repo.openeuler.org/openEuler-24.03-LTS/virtual_machine_img/aarch64/openEuler-24.03-LTS-aarch64.qcow2.xz
xz -d openEuler-24.03-LTS-aarch64.qcow2.xz
cp /usr/share/qemu-efi-aarch64/QEMU_EFI.fd .
```

### 2-3 启动

下载[start.sh](qemu_openeuler/aarch64/start.sh)文件

```shell
chmod +x start.sh
./start.sh
```

还需要修改/etc/ssh/sshd_config，然后在host生成公密钥，把公钥添加到qemu的/root/.ssh/authorized_keys中，然后就可以[连接](qemu_openeuler/aarch64/ssh.sh)了。
