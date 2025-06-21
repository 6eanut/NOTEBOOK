# QEMU AMD64 Ubuntu

> 在日常学习中，会遇到这样一种情况，你读到一篇几年前的论文，然后想要复现一下论文里实现的工具，但是你现在用的环境比较新，而论文是几年前发布的，虽然其开源了代码，但是环境配置很麻烦，因为工具用到的一些文件可能在新版本中弃用了，所以复现工具之前，拿到一个和当时类似的环境是很重要的。

参考资料：[https://documentation.ubuntu.com/public-images/public-images-how-to/use-local-cloud-init-ds/#use-local-cloud-init-ds](https://documentation.ubuntu.com/public-images/public-images-how-to/use-local-cloud-init-ds/#use-local-cloud-init-ds)

为什么复现环境要用qemu，不用docker？因为docker默认是把文件放在主机的/var下面，但是/var空间有限，所以采用qemu。

整个过程比较简单，这里以ubuntu16.04为例，如下：

```shell
sudo apt update
sudo apt install --yes cloud-image-utils qemu-system-x86

cat > user-data.yaml <<EOF
#cloud-config
password: password
chpasswd:
  expire: False
ssh_pwauth: True
ssh_authorized_keys:
  - ssh-rsa AAAA...UlIsqdaO+w==
EOF

echo "instance-id: $(uuidgen || echo i-abcdefg)" > my-meta-data.yaml

cloud-localds my-seed.img user-data.yaml my-meta-data.yaml

wget https://cloud-images.ubuntu.com/xenial/current/xenial-server-cloudimg-amd64-disk1.img
qemu-system-x86_64  \
  -cpu host -machine type=q35,accel=kvm -m 2048 \
  -nographic \
  -snapshot \
  -netdev id=net00,type=user,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net00 \
  -drive if=virtio,format=qcow2,file=xenial-server-cloudimg-amd64-disk1 \
  -drive if=virtio,format=raw,file=my-seed.img \
  2>&1 | tee vm.log
```

几点说明：

* 进入qemu之后，用户名是 `ubuntu`，密码是 `password`；
* ssh连接方式：`ssh -o "StrictHostKeyChecking no" ubuntu@0.0.0.0 -p 2222`;
* 如果想要装其他版本，访问：[https://cloud-images.ubuntu.com](https://cloud-images.ubuntu.com)；
