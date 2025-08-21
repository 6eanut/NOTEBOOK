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

---

# 20250820在X86_64上搭建QEMU RISC-V环境，Linux内核自己编译

```shell
**工作目录**
mkdir riscv64-linux
cd riscv64-linux

**交叉工具链**
wget https://mirror.iscas.ac.cn/riscv-toolchains/release/riscv-collab/riscv-gnu-toolchain/LatestRelease/riscv64-glibc-ubuntu-22.04-gcc-nightly-2025.05.30-nightly.tar.xz
tar -xvf riscv64-glibc-ubuntu-22.04-gcc-nightly-2025.05.30-nightly.tar.xz
--修改环境变量
riscv64-unknown-linux-gnu-gcc -v

**编译Linux内核**
git clone --branch v6.16 git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- defconfig
--打开以下配置项
--注意CONFIG_KCOV_ENABLE_COMPARISONS 和 CONFIG_KASAN_INLINE没法开，否则qemu无法启动
CONFIG_KCOV=y
CONFIG_KCOV_INSTRUMENT_ALL=y
CONFIG_DEBUG_FS=y
CONFIG_DEBUG_KMEMLEAK=y
CONFIG_DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT=y
CONFIG_KALLSYMS=y
CONFIG_KALLSYMS_ALL=y
CONFIG_NAMESPACES=y
CONFIG_UTS_NS=y
CONFIG_IPC_NS=y
CONFIG_PID_NS=y
CONFIG_NET_NS=y
CONFIG_CGROUP_PIDS=y
CONFIG_MEMCG=y
CONFIG_USER_NS=y
# CONFIG_RANDOMIZE_BASE is not set
CONFIG_KASAN=y
CONFIG_FAULT_INJECTION=y
CONFIG_FAULT_INJECTION_DEBUG_FS=y
CONFIG_FAULT_INJECTION_USERCOPY=y
CONFIG_FAILSLAB=y
CONFIG_FAIL_PAGE_ALLOC=y
CONFIG_FAIL_MAKE_REQUEST=y
CONFIG_FAIL_IO_TIMEOUT=y
CONFIG_FAIL_FUTEX=y
CONFIG_LOCKDEP=y
CONFIG_PROVE_LOCKING=y
CONFIG_DEBUG_ATOMIC_SLEEP=y
CONFIG_PROVE_RCU=y
CONFIG_DEBUG_VM=y
CONFIG_REFCOUNT_FULL=y
CONFIG_FORTIFY_SOURCE=y
CONFIG_HARDENED_USERCOPY=y
CONFIG_LOCKUP_DETECTOR=y
CONFIG_SOFTLOCKUP_DETECTOR=y
CONFIG_HARDLOCKUP_DETECTOR=y
CONFIG_BOOTPARAM_HARDLOCKUP_PANIC=y
CONFIG_DETECT_HUNG_TASK=y
CONFIG_WQ_WATCHDOG=y
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- olddefconfig
make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- -j

**OpenSBI**
git clone https://github.com/riscv/opensbi
cd opensbi
make CROSS_COMPILE=riscv64-unknown-linux-gnu- PLATFORM_RISCV_XLEN=64 PLATFORM=generic

**buildroot**
wget https://buildroot.org/downloads/buildroot-2025.02.5.tar.gz
tar -xvf buildroot-2025.02.5.tar.gz
make qemu_riscv64_virt_defconfig
make menuconfig
--打开以下选项
    Target packages
	    Networking applications
	        [*] iproute2
	        [*] openssh
    Filesystem images
                ext2/3/4 variant - ext4
	        exact size - 1g
--关闭以下选项
    Kernel
	    Linux Kernel
make
--为output/target/etc/fstab文件添加下面这一行
debugfs	/sys/kernel/debug	debugfs	defaults	0	0
--为output/target/etc/ssh/sshd_config中替换下面几行
PermitRootLogin yes
PasswordAuthentication yes
PermitEmptyPasswords yes
make

**qemu启动**
mkdir qemu
cd qemu
qemu-system-riscv64 \
	-machine virt \
	-nographic \
	-bios /home/jiakai/riscv64-linux/opensbi/build/platform/generic/firmware/fw_jump.bin \
	-kernel /home/jiakai/riscv64-linux/linux/arch/riscv/boot/Image \
	-append "root=/dev/vda ro console=ttyS0" \
	-object rng-random,filename=/dev/urandom,id=rng0 \
	-device virtio-rng-device,rng=rng0 \
	-drive file=/home/jiakai/riscv64-linux/buildroot-2025.02.5/output/images/rootfs.ext2,if=none,format=raw,id=hd0 \
	-device virtio-blk-device,drive=hd0 \
	-netdev user,id=net0,host=10.0.2.10,hostfwd=tcp::10022-:22 \
	-device virtio-net-device,netdev=net0

**syzkaller**
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
make TARGETOS=linux TARGETARCH=riscv64
```
