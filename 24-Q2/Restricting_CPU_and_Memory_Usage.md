# Restricting CPU and Memory Usage

最近在鲲鹏920上跑tensorflow的models，虽然内存16G、16核，但是跑一些大模型时会出现cpu和内存的利用率较高，达到百分之八十以上，然后总共110个steps，运行到60左右就停掉了。

benchmark的终端还是显示未终止，但是top查看进程会发现已经断掉了，所以想着限制一下一个进程运行时的cpu和内存的利用率。

## 限制CPU

原命令如下：

```
python3 /home/6eanut/tensorflow-test/benchmarks/perfzero/lib/benchmark.py --git_repos="https://github.com/tensorflow/models.git;benchmark" --python_path=models --gcloud_key_file_url="" --benchmark_methods=official.benchmark.resnet_ctl_imagenet_benchmark.Resnet50CtlBenchmarkSynth.benchmark_1_gpu
```

限制CPU后的命令如下：

```
taskset -c 0-9 python3 /home/6eanut/tensorflow-test/benchmarks/perfzero/lib/benchmark.py --git_repos="https://github.com/tensorflow/models.git;benchmark" --python_path=models --gcloud_key_file_url="" --benchmark_methods=official.benchmark.resnet_ctl_imagenet_benchmark.Resnet50CtlBenchmarkSynth.benchmark_1_gpu
```

**`taskset`命令可以将进程绑定到特定的CPU核心上**

## 限制内存

限制内存后的命令如下：

```
ulimit -v 10000000 && python3 /home/6eanut/tensorflow-test/benchmarks/perfzero/lib/benchmark.py --git_repos="https://github.com/tensorflow/models.git;benchmark" --python_path=models --gcloud_key_file_url="" --benchmark_methods=official.benchmark.resnet_ctl_imagenet_benchmark.Resnet50CtlBenchmarkSynth.benchmark_1_gpu
```

---

# 内存不够用怎么办(20250414)

## zRAM

zRAM的全称是compressed RAM，是Linux内核的一个模块。用于将部分内存作为压缩的swap分区使用，与磁盘无关。

应用程序申请内存->内核发现物理内存不足->将冷内存页压缩存入zRAM->后续访问时解压

```shell
sudo swapon --show								# 查看当前swap分区的情况
sudo modprobe zram								# 内核加载zram模块
echo "zram" | sudo tee /etc/modules-load.d/zram.conf				# Linux在启动时自动扫描并加载/etc/modules-load.d下的模块
echo "4G" | sudo tee /sys/block/zram0/disksize					# 设置zram的大小
sudo zramctl									# 查看zram的设置
sudo mkswap /dev/zram0								# 将zram设置为swap
sudo swapon /dev/zram0 -p 100							# 启用zram
sudo swapon --show								# 查看当前swap分区的情况
free -h										# 查看memory的情况
echo "/dev/zram0 none swap defaults,pri=100 0 0" | sudo tee -a /etc/fstab	# 将zram设备设置为系统启动时自动挂载为swap
```

## 扩大swap

可以通过创建新的swap来扩大swap分区

```shell
sudo dd if=/dev/zero of=/swapfile bs=1G count=4 status=progress		# 每次读写块大小为1GB，共四个块
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
swapon --show
echo "/swapfile none swap defaults 0 0" | sudo tee -a /etc/fstab
```
