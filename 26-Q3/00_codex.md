# Codex

> Claude的Model和Harness都很厉害，两者合在一起最厉害
>
> Claude的Model太贵了，不用也行，Harness不要钱，所以接一个中转站，用便宜一些的Model
>
> 便宜Model+Claude Harness开销也不小，每天20+美金
>
> ChatGPT Plus一个月才二百人民币，这么比下来性价比很高(虽然五小时和一周有限额)
>
> 所以Codex和Claude Harness+Cheap Model二者混合用，生产力/代价最大化

Codex充值找代充即可

Codex CLI是在服务器上用

需要设置下面两个内容(在本地笔记本)：

```
ssh -L 1455:127.0.0.1:1455 server
ssh -R 17890:127.0.0.1:7890 server
```

远程连接到服务器：

```
ssh server
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
npm i -g @openai/codex
codex login
# 接下来会弹出一个链接，然后复制到本地笔记本的浏览器中进行登录即可
```
