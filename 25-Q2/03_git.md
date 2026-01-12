# git Startup

```txt
# 设置本地仓库url
git remote set-url origin url
# 更新远程仓库信息到本地
git fetch origin
# 查看当前仓库位于哪个commit
git rev-parse HEAD
# 查看两个commit间的差异
git diff commit1..commit2 > update.patch
```

---

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

当社区内的资料不够多，并且其他网站的材料也没多少时，邮件联系maintainer是个不错的方法，有时人们并不会把自己的联系邮件放在profile里面，但是git提供了很好的帮助，我们能找到做出commit最多的几个用户。

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

## 7 只clone仓库中的指定路径下的文件

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

# 推送新名称的分支到远程
git push origin dev-main

# 删除远程的旧分支
git push origin --delete dev
```

## 11 查看历史commit修改内容

```shell
git log
git show commit_id
```

## 12 git email

```shell
# 拉源码
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux

# 配个人信息
git config --local user.name "Your Name"
git config --local user.email "your_email@example.com"

# 对源码做修改并且附上相关信息(第一行为标题，之后需要空一行)
vim drivers/.../xxx.c
git add drivers/.../xxx.c
git commit -s
git commit --amend

# 生成补丁
git format-patch -1
./scripts/checkpatch.pl 0001-fix-driver-comment.patch
```

## 13 删除已经提交到线上仓库的commit

```shell
# 回退到倒数第3个 commit（即撤销最近的2个）
git reset --hard HEAD~2
git push origin <分支名> --force
```

## 14 同步社区主仓库到本地fork仓库

```shell
# 将 6098a8e (你的修改) 直接接到 15f6fd0 (官方最新提交) 之上
git rebase --onto 15f6fd0844 86fe55eb76 kvm_riscv
现在有三个commit号，commit1表示你fork社区主仓库时的commit，commit2表示你在commit1的基础上添加自己修改后的commit，commit3表示社区主仓库从commit1更新到commit3的commit
git rebase --onto commit3 commit1 <q分支名>

-----------------------------------------------------------

# 这么做会让你把基于老commit做的本地修改应用到基于新commit做的修改
git remote add upstream https://github.com/google/syzkaller.git
git fetch upstream
git rebase upstream/master
```

## 15 查看修改

```shell
# 对于git status得到的修改，想要查看单个文件做了哪些修改
git diff file.c
```

## 16 查看远程仓库有哪些分支并且删除远程仓库的分支

```shell
git branch -r

git push origin --delete branch_name
```
