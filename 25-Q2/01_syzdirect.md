# Syzdirect

> Syzdirect是一个基于Syzkaller实现的Linux Kernel定向灰盒模糊测试(DGF)工具，其能通过给定的内核commit，来对该特定位置进行模糊测试。

* 论文阅读：[SyzDirect: Directed Greybox Fuzzing for Linux Kernel](https://6eanut.github.io/PaperReading/SyzDirect/note.html)
* syzdirect和syzkaller的初对比：[目录结构的差异](syzdirect/00-note.md)
* 语料库：[距离衡量程序的优先级](syzdirect/01-Corpus.md)
* 选择表：[新添系统调用对信息](syzdirect/02-ChoiceTable.md)
* 小总结：[从三个数据结构来总结](syzdirect/04-check.md)
* 系统调用对：[初始化及作用](syzdirect/05-CallPairMap.md)
