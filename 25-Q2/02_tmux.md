# tmux Startup

本文记录tmux的使用方法，为了方便查阅：

## 0 创建session

```shell
# 启动一个
tmux
# 指定session名字
tmux new -s session-name
```

## 1 关闭session

```shell
exit
ctrl + d
```

## 2 查询session

```shell
tmux ls
```

## 3 分离session

```shell
# 保持后台运行
ctrl + b, d
```

## 4 重连分离后的session

```shell
tmux attach -t session-name
```

## 5 在session中创建新窗口

```shell
ctrl + b, c
```

## 6 在session中切换窗口

```shell
ctrl + b, 窗口编号
```
