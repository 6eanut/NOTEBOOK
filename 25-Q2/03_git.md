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

## 4 合并commit

自己在开发的时候会很随意的commit、push，但是当想要去给社区提交pr的时候，一个pr里面含有多个commit，而且附带的信息也很乱，这时，就需要把多个commit合并为1个，然后附上有意义的信息。

```shell
# 首先确定要合并的commit的数量，[N]是数量
git rebase -i HEAD~[N]

# 随后会进入到一个编辑器，如下，将第一个 commit 保留为 pick，后续的改为 squash 或 s
pick abc123 第一个commit
squash def456 第二个commit
squash ghi789 第三个commit

# 修改上一个已经commit的信息
git commit --amend -m "新的提交信息"

# 强制推送
git push -f
```

## 5 联系maintainer

当社区内的资料不够多，并且其他网站的材料也没多少时，邮件联系maintainer是个不错的方法，有时人们并不会把自己的联系邮件放荡profile里面，但是git提供了很好的帮助，我们能找到做出commit最多的几个用户。

```shell
git log --pretty="%ae" | sort | uniq -c | sort -nr | head -6
```

## 6 修改上一次commit的内容

```
# 在本地做修改

# 修改之后保存
git add .

# 覆盖上一次的提交
git commit --amend  # 保持相同的提交信息，或修改它

# 强制推送
git push origin/<branch-name> --force

# 此时，在另一台留有老commit的设备上git pull不能用了
git fetch origin
git reset --hard origin/<branch-name>
```

## 7 只clone仓库中的指定路径下的文集

```shell
git clone --filter=blob:none --no-checkout https://gitee.com/peeanut/kernels_repo.git
cd kernels_repo
git sparse-checkout init --cone
git sparse-checkout set kernel_v4.17 static_analysis/kernel_v4.17
git checkout master  # 或你的目标分支（如 main）
```

## 8 源码编译安装git

```shell
# https://github.com/git/git/tags 去下载想要的版本的git源码
make prefix=pathto/git-install all
make prefix=pathto/git-install install
```

## 9 创建新分支

```shell
git checkout -b new_branch
git add .
git commit -m "message"
git push -u origin new_branch
```

## 10 给已有分支重命名

```shell
# 先切换到要重命名的 dev 分支
git checkout dev

# 重命名当前分支为 dev-main (或其他名字)
git branch -m dev-main
```
