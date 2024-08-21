# Install the rpm package to the user path

在日常，往往是多个人使用同一台服务器，如果自己想要dnf install或者将某个自己构建的rpm安装到系统的root里，则会有可能影响其他用户的使用，所以这里提供一种将rpm包安装到个人用户下的方法。

## 获取rpm包

如果想要通过dnf install来安装某个rpm包，可以先通过dnf download下载rpm包。

## 安装到个人目录

```
rpm2cpio name.rpm | cpio -idmv
mkdir -p ~/local/namePath
mv ./usr/* ~/local/namePath
vim ~/.bashrc
# 把路径添加到环境变量中
```
