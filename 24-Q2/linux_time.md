# Set the linux system time to the local time in China

```
sudo timedatectl set-timezone Asia/Shanghai
```

```
[root@jiakai-openeuler-01 ~]# date
Thu Jun 27 06:37:30 PM CST 2024
```

---

```
# 另一种方法
sudo yum install -y ntpdate
sudo ntpdate pool.ntp.org
```
