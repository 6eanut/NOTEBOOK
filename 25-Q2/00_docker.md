# Docker Startup

以下是一些常见操作，便于查阅：

## 安装docker

```shell
sudo apt install docker docker.io
```

## 添加用户到docker组，否则无法使用

```shell
sudo usermod -aG docker $USER
```

## 查看docker版本

```shell
docker --version
```

## 查看容器

```shell
docker ps -a
```

## 查看镜像

```shell
docker images
```

## 登录dockerhub

```shell
docker login
```

## 拉取镜像

```shell
docker pull repository:tag
```

## 更新镜像仓库名和标签

```shell
docker tag "IMAGE ID" repository:tag
```

## 从镜像创建容器

```shell
docker run -it [--cpus="8" --memory="32g"] repository:tag /bin/bash
```

## 容器更名

```shell
docker rename oldname newname
```

## 启动已创建的容器

```shell
docker start container_name
```

## 进入已经启动的容器

```shell
docker exec -it [-u user_name] container_name /bin/bash
```

## 查看docker磁盘占用情况

```shell
docker system df
```

## 文件传输

```shell
docker ps -a					# 查询container_name
docker inspect -f '{{.Id}}' container_name	# 查询container_id
docker cp hostPath container_id:containerPath
```

## 将容器提交为新镜像

```shell
docker commit container_name dockerhub_name/repo:tag
```

## 为镜像打标签

```shell
docker tag image_name:tag dockerhub_name/repo:tag
```

## 推送镜像到dockerhub

```shell
docker push dockerhub_name/repo:tag
```

## 删除指定名称和标签的镜像

```shell
docker rmi image_name:tag
```

## 将镜像压缩为文件

```shell
# 语法：docker save -o <输出文件名.tar> <镜像名:标签>
docker save -o my_image.tar nginx:latest
# 语法：docker load -i <文件名.tar>
docker load -i my_image.tar
```
