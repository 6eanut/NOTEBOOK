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

## 7 给session起名字

```shell
tmux rename-session -t <old_name> <new_name>
```

## 8 在session内创建pane

```shell
# 水平分割
ctrl + b, "
# 垂直分割
ctrl + b, %
# 按顺序切换到下一个pane
ctrl + b, o
# 方向键切换pane
ctrl + b, 方向键
# 按编号切换pane
ctrl + b, q
```

## 9 tmux配置文件

```shell
# 修改.tmux.conf文件
set -g default-terminal "screen-256color"
set-option -g mouse on
set-option -g renumber-windows on

set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-pain-control'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-yank'

# Initialize TMUX plugin manager (keep this line at the very bottom of tmux.conf)
run '~/.tmux/plugins/tpm/tpm'
```

## 10 显示CPU/Memory利用率

```shell
# 可以用htop/top来看，但是那不方便，可以把这个功能嵌入到tmux里面
set -g status-interval 2
set -g status-right "#[fg=green]CPU: #(bash -c 'read a b c d e f g h i j < /proc/stat; sleep 0.3; read a2 b2 c2 d2 e2 f2 g2 h2 i2 j2 < /proc/stat; idle=$((d+e)); idle2=$((d2+e2)); total=$((b+c+d+e+f+g+h+i)); total2=$((b2+c2+d2+e2+f2+g2+h2+i2)); echo $(( (100*( (total2-total)-(idle2-idle) )) / (total2-total) ))%') #[fg=cyan]| MEM: #(free -h | awk '/Mem:/ {print $3 \"/\" $2}') #[fg=yellow]| %H:%M "
```

## 11 pane的同步

```shell
ctrl + b, :set synchronize-panes
ctrl + b, :set synchronize-panes off
```
