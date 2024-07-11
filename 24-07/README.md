# 24-07目录说明

## 0-笔记

[Caffe on Aarch64](caffe-aarch64.md)

[Tensorflow on Aarch64](tensorflow-aarch64.md)

## 1-image

笔记中所涉及到的图片在此文件夹中。

## 2-caffe

### deploy

包含了50个用于 `caffe time`的模型的 `deploy.prototxt`文件。

使用方法：

```
wget https://raw.githubusercontent.com/6eanut/NOTEBOOK/main/24-07/caffe/shell/caffe-test.sh
chmod +x caffe-test.sh
./caffe-test.sh
```

### makefiles

包含在`riscv64`和`aarch64`上编译caffe时所需要的`Makefile.config`。

### shell

[性能测试脚本](caffe/shell/perf_information_get.sh)

aarch64上caffe的[自动化构建脚本](caffe/shell/caffe-aarch64.sh)

riscv64上caffe的[自动化构建脚本](caffe/shell/caffe-riscv64.sh)

caffe的models[自动化测试脚本](caffe/shell/caffe-test.sh)
