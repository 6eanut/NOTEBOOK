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
b
```

**`taskset`命令可以将进程绑定到特定的CPU核心上**

## 限制内存

限制内存后的命令如下：

```
ulimit -v 10000000 && python3 /home/6eanut/tensorflow-test/benchmarks/perfzero/lib/benchmark.py --git_repos="https://github.com/tensorflow/models.git;benchmark" --python_path=models --gcloud_key_file_url="" --benchmark_methods=official.benchmark.resnet_ctl_imagenet_benchmark.Resnet50CtlBenchmarkSynth.benchmark_1_gpu
```
