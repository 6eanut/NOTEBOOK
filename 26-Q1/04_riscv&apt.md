# RISC-V & APT

在[03 QEMU OpenEuler(RISC-V Aarch64)](25-Q1/03_qemu_openeuler.md)这篇里面记录了如何自己编译一个RISC-V Linux，以及如何用QEMU启动它。

但是，如果按照那个方法，进入系统是没有APT/DNF这样的包管理器的，因为文件镜像用的是BUILDROOT。

那我想用APT/DNF装一些包咋办。随着RISC-V的快速发展，越来越多的发行版Linux开始支持RISC-V，就有了下面的方法。

[https://github.com/carlosedp/riscv-bringup](https://github.com/carlosedp/riscv-bringup)

用下面的命令来启动：

```
qemu-system-riscv64 \
  -cpu max \
  -M virt,aclint=on,aia=aplic-imsic,aia-guests=3,iommu-sys=on,acpi=on \
  -m 4096 \
  -smp 4 \
  -chardev socket,id=SOCKSYZ,server=on,wait=off,host=localhost,port=64252 \
  -mon chardev=SOCKSYZ,mode=control \
  -display none \
  -serial stdio \
  -no-reboot \
  -name VM-0 \
  -device virtio-rng-pci \
  -machine virt \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,restrict=on,hostfwd=tcp:127.0.0.1:18754-:22 \
  -device virtio-blk-device,drive=hd0 \
  -drive file=path/rootfs.img,if=none,format=raw,id=hd0 \
  -snapshot \
  -kernel pathto/linux/arch/riscv/boot/Image \
  -append "root=/dev/vda console=ttyS0 rw earlycon=sbi earlyprintk panic_on_warn=1 init=/bin/bash" \
  2>&1 | tee vm.log 
```

其中Image可以通过编译Linux获得，下面是获得rootfs.img的方法：

```
# 1. 下载并解压 rootfs
wget https://github.com/carlosedp/riscv-bringup/releases/download/v1.0/debian-sid-riscv64-rootfs-20200108.tar.bz2
mkdir rootfs
sudo tar -xpf debian-sid-riscv64-rootfs-20200108.tar.bz2 -C rootfs

# 2. 创建一个 ext4 镜像文件（比如 2GB）
dd if=/dev/zero of=rootfs.img bs=1M count=2048
mkfs.ext4 rootfs.img

# 3. 把 rootfs 内容写入镜像
mkdir mnt
sudo mount rootfs.img mnt
sudo cp -a rootfs/* mnt/
sudo umount mnt
```

注意，在前面的启动命令里有init=/bin/bash，这是因为测试时ttys0一直过不去，所以选择这个方法绕过去。

如果绕过去的话，进去是需要手动开ssh服务的，然后就可以ssh/scp了。

```
ip addr add 10.0.2.15/24 dev eth0
ip route add default via 10.0.2.2
mkdir -p /run/sshd
/usr/sbin/sshd
ssh-keygen -A
/usr/sbin/sshd
```
