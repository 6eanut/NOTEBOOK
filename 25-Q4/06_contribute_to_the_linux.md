# Contribute to the Linux Community

Linus所维护的Linux仓库是mainline，贡献代码一般是基于某个特定的子系统做修改，然后向该子系统的maintainer所维护的tree提patch。

* repositories index：[https://git.kernel.org/](https://git.kernel.org/)
* public-inbox listing：[https://lore.kernel.org/](https://lore.kernel.org/)

这里以riscv为例：

## 1 本地修改

首先拉取源码并切换到最新的tag：

```shell
git clone https://kernel.googlesource.com/pub/scm/linux/kernel/git/riscv/linux.git riscv-for-linus-6.18-rc6
cd riscv-for-linus-6.18-rc6
git checkout riscv-for-linus-6.18-rc6
```

然后做相应的修改，并制作补丁：

```shell
vi arch/riscv/kernel/tests/kprobes/test-kprobes-asm.S
git add arch/riscv/kernel/tests/kprobes/test-kprobes-asm.S
git checkout -b local-fix-test_kprobes
git commit -s
# 此时需要写上标题、修改的原因以及修改的内容
git format-patch -1
```

如果这是你的v3版patch，那么可能需要进入到patch文件里面去做修改，记录这一版相较于前一版做了哪些修改，比如：

```patch
Signed-off-by: Jiakai Xu <jiakaiPeanut@gmail.com>
---

Changes since v2:
 * Fixed line-wrapping issues to ensure patch applies cleanly.
 * Formatting improvements in the commit message.
Thanks to Jonathan Corbet for pointing out this additional correction.

Changes since v1:
 * Added the second fix for the documentation comment line.
Thanks to Randy Dunlap for pointing out this additional correction.

References:
 * [PATCH] Documentation/admin-guide: fix typo in cscope command example
https://lore.kernel.org/linux-doc/6017104c-740d-43db-bc53-58751ec57282@infradead.org/T/#t
 * [PATCH v2] Documentation/admin-guide: fix typo and comment in cscope example
https://lore.kernel.org/linux-doc/871plv5mlf.fsf@trenco.lwn.net/T/#m10f8ec032dd57eaf7388939da3722c9f4b599b33
 Documentation/admin-guide/workload-tracing.rst | 10 +++++-----
 1 file changed, 5 insertions(+), 5 deletions(-)

diff --git a/Documentation/admin-guide/workload-tracing.rst b/Documentation/admin-guide/workload-tracing.rst
index d6313890ee41..35963491b9f1 100644
--- a/Documentation/admin-guide/workload-tracing.rst
+++ b/Documentation/admin-guide/workload-tracing.rst
@@ -196,11 +196,11 @@ Let’s checkout the latest Linux repository and build cscope database::
   cscope -R -p10  # builds cscope.out database before starting browse session
   cscope -d -p10  # starts browse session on cscope.out database
 
-Note: Run "cscope -R -p10" to build the database and c"scope -d -p10" to
-enter into the browsing session. cscope by default cscope.out database.
-To get out of this mode press ctrl+d. -p option is used to specify the
-number of file path components to display. -p10 is optimal for browsing
-kernel sources.
+Note: Run "cscope -R -p10" to build the database and "cscope -d -p10" to
+enter into the browsing session. cscope by default uses the cscope.out
+database. To get out of this mode press ctrl+d. -p option is used to
+specify the number of file path components to display. -p10 is optimal
+for browsing kernel sources.
 
 What is perf and how do we use it?
 ==================================
-- 
2.34.1
```

修改完之后，需要运行scripts下的脚本来获取maintainer/supporter的邮箱以及检查patch：

```shell
./script/checkpatch.pl xxx.patch
./scripts/get_maintainer.pl arch/riscv/kernel/tests/kprobes/test-kprobes-asm.S
```

## 2 邮件发送

不要采用客户端/网页版发送，用git的send-email来发，这里以gmail为例：

```shell
git config --local user.email "jiakaiPeanut@gmail.com"
git config --local user.name "Jiakai Xu"
git config --local sendemail.smtpserver smtp.gmail.com
git config --local sendemail.smtpuser "jiakaiPeanut@gmail.com"
git config --local sendemail.smtpserverport 587
git config --local sendemail.smtpencryption tls
git config --local sendemail.smtppass "xxxx yyyy zzzz nnnn"
```

其实user.email和user.name应该在commit之前就设置好。

smtppass通过[https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)来设置。

然后发送：

```shell
git send-email \
  xxx.patch \
  --smtp-debug=1 \
  --to "linux-doc@vger.kernel.org" \
  --to "linux-kernel@vger.kernel.org" \
  --cc "Jonathan Corbet <corbet@lwn.net>" \
  --cc "Randy Dunlap <rdunlap@infradead.org>"
```

---

很多邮件方面的沟通模式还需要多学习，后面会进行更新。
