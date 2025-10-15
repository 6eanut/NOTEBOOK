# Python-venv

venv:**V**irtual **ENV**ironment

## 创建venv

python -m venv /home/venv/venv-test

```shell
[venv@jiakai-openeuler-01 venv-test]$ ls
bin  include  lib  lib64  pyvenv.cfg
[venv@jiakai-openeuler-01 venv-test]$ pwd
/home/venv/venv-test

```

## 切换venv

source /home/venv/venv-test/bin/activate

```shell
[venv@jiakai-openeuler-01 ~]$ source /home/venv/venv-test/bin/activate
(venv-test) [venv@jiakai-openeuler-01 ~]$
```

## 退出venv

deactivate

```shell
(venv-test) [venv@jiakai-openeuler-01 ~]$ deactivate
[venv@jiakai-openeuler-01 ~]$

```

## 感想

像比venv功能更强大的有比较知名的anaconda、conda、pyenv等等，他们不仅可以创建虚拟环境，还可以指定虚拟环境的python版本，不像venv，只能用base环境的python版本。但是个人感觉如果不需要切换python版本的话，用venv就够了。

比如在一个venv下，可以从零开始搭建环境，很方便，心里很清透。搭建失败了或者不用了，可以直接删掉。

最近在鲲鹏920上先后测试了caffe和tensorflow，我的做法就是用root用户先后创建了两个用户，分别用了测试caffe和tensorflow，这样也行，但是感觉这种抽象层次太高了，完全可以在一个用户下，抽象出两个venv，然后在不同的venv里面做测试，感觉这样更舒服一点。

但是，如果像之前移植opengauss的项目，那最好还是用anaconda，因为它可以切换python版本。所以到底用不用anaconda，用不用venv，还是要看具体情况。

接触项目越多，越发现版本的重要性，还是要先弄清楚版本，然后会更方便。

---

# pyenv(20250320)

[pyenv](https://github.com/pyenv/pyenv)是一个python的版本管理工具，支持下载多个版本的python并进行自由切换，其是一个开源项目。

## 安装

```shell
# 装依赖包
sudo apt install -y curl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libffi-dev tk-dev libncurses5-dev libncursesw5-dev
sudo dnf install -y gcc make zlib-devel bzip2-devel openssl-devel libffi-devel readline-devel sqlite-devel xz-devel
# 安装pyenv
curl -fsSL https://pyenv.run | bash
# 配置pyenv
export PATH=/home/$(whoami)/.pyenv/bin:$PATH
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

## 使用

```shell
# 查看当前系统安装的python版本有哪些
pyenv versions
# 安装某一版本的python
pyenv install 3.x.x
# 卸载某一版本的python
pyenv uninstall 3.x.x
# 切换到某一版本的python(整个用户空间)
pyenv global 3.x.x
# 切换到某一版本的python(当前目录及其子目录)
pyenv local 3.x.x
```

## 感想

pyenv是一个很强大的工具，它甚至提供在不同的工作目录下使用不同python版本的功能。

# 如何管理本地的python2和python3(20251015)

这里借助update-alternatives工具来实现：

```shell
# 查看当前可用的python
update-alternatives --list python

# 配置
update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2

# 选择当前python指定的版本
update-alternatives --config python
```
