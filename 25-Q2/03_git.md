# git Startup

如果一个人有多个github账号，在本地会对不同账号下的不同仓做修改并commit和push，本地机该如何做配置？本篇演示环境为Windows 11。

## 0 为新账号生成新的ssh公密钥

```shell
ssh-keygen -t ed25519 -C "{useremail}"
```

然后，将生成的ssh公钥放在github下，并在本地的config文件中添加如下内容：

```config
Host {username}
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_{username}
```

## 1 验证ssh可连接性

```shell
ssh -vT git@{username}
```

然后应该能看到 `Hi {username}!`的打印信息。

## 2 clone仓库

```shell
# 通过ssh来clone仓库
git clone git@github.com:repoPath/reponame.git
cd reponame
# 配置该仓的name和email
git config --local user.name "{username}"
git config --local user.email "{useremail}"
# 检查是否配置成功
git config user.name
git config user.email
```

## 3 修改url

```shell
# 查看当前url
git remote -v
# 修改url
git remote set-url origin git@{username}:repoPath/reponame.git
```

在此之后，就可以做代码修改，而后add、commit和push了。

## 4 联系maintainer

当社区内的资料不够多，并且其他网站的材料也没多少时，邮件联系maintainer是个不错的方法，有时人们并不会把自己的联系邮件放荡profile里面，但是git提供了很好的帮助，我们能找到做出commit最多的几个用户。

```shell
git log --pretty="%ae" | sort | uniq -c | sort -nr | head -6
```
