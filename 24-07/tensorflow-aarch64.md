# Tensorflow on Aarch64

## 1-环境搭建

测试tensorflow中的models需要安装 `tensorflow`、`tensorflow-text`、`tf-models-official`，为了方便管理python环境，所有的操作都在venv中进行。

### venv

python虚拟环境的基本使用方法如下：

```
//创建
python -m venv /home/tf-test/venv00
//激活
source /home/tf-test/venv00/bin/activate
//退出
deactivate
```

激活后，出现如下图红圈所示，即为进入python虚拟环境，然后就可以开始操作了。

![1720168211338](image/tensorflow-aarch64/1720168211338.png)

### tensorflow

* github：https://github.com/tensorflow/tensorflow
* version：2.15.0

* method：pip install tensorflow==2.15.0

tensorflow是pypi中的一个包，可以直接通过`pip install tensorflow==2.15.0`来安装。

### tensorflow-text

* github：https://github.com/tensorflow/text
* version：2.15.0

* method：build from source

#### step1-bazel 6.1.0

编译tensorflow-text 2.15.0需要bazel 6.1.0。

```
//从github下载bazel 6.1.0的release
wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-linux-arm64
//修改权限
chmod +x bazel-6.1.0-linux-arm64
//修改名字
mv bazel-6.1.0-linux-arm64 bazel
//初始化
./bazel
//将bazel可执行文件的路径添加到PATH环境变量中
export PATH="$PATH:PathToBazel"
```

运行`bazel --version`命令查看是否安装并配置成功。

![1720170318183](image/tensorflow-aarch64/1720170318183.png)

#### step2-text 2.15.0

tensorflow-text的github仓库有从源码构建的[步骤](https://github.com/tensorflow/text?tab=readme-ov-file#build-from-source-steps)，具体如下：

```
//从github下载text源码
git clone https://github.com/tensorflow/text
cd text
//切换到2.15分支
git checkout 2.15
//运行脚本，构建wheel
./oss_scripts/run_build.sh
//安装text 2.15.0
pip install tensorflow_text-2.15.0-cp311-cp311-linux_aarch64.whl
```

### tensorflow-models

* github：https://github.com/tensorflow/models
* version：2.15.0

* method：pip install tf-models-official==2.15.0

tensorflow-models是pypi中的一个包，可以直接通过 `pip install tf-models-official==2.15.0`来安装。

### 环境检查

通过`pip show tensorflow tensorflow-text tf-models-official`命令检查环境是否搭建完毕。

![1720167945517](image/tensorflow-aarch64/1720167945517.png)

## 2-问题解决

## 3-用例运行
