# 啥是X86？

> 这是一篇零碎生活记录

---

今天是阳历2025年2月19日。

在2009年，我爸妈买了一台联想的台式电脑，我也不知道为了啥要买。我在上面玩了大鱼吃小鱼、极品飞车、红警、植物大战僵尸、奥比岛、洛克王国、赛尔号、功夫派、QQ飞车、穿越火线、实况足球······直到有一天，当我按下主机的开机键时，会发出“嗡嗡嗡”的声音，然后安静三四秒，再开始“嗡嗡嗡”，这样一直重复。当时请了修电脑的人来看，说是主机带不动Windows 7，降成Windows XP就好了，后来我也不知道为啥一直没修好。

刚上大学时，我爸妈给我买了一台小新的笔记本电脑，因为大学专业是计算机。我在上面看B站、学编程。每逢寒暑假回家，都会下载实况足球和足球经理，和许佳琦一起玩。直到有一天，每当游戏打开不到半个小时，笔记本就烫的不行，我感觉磕上鸡蛋真能熟，即使寒假也不例外。当时就想出了一个办法，拿家里的电扇照着笔记本吹风，这样就好多了。

这个寒假，我网购了螺丝刀们，打算拆开笔记本后壳看看，清理下风扇上的灰尘。果真，灰尘很多，严重影响了散热，清理过后，笔记本明显没之前那么烫了。

---

既然都动工拆笔记本了，那就顺带看看老电脑吧。

我把显示器、鼠标、键盘、音响都从主机上拔了下来，只把主机连上电源，按开机键，发现还是之前的老毛病。于是，我卸掉了主机侧面的壳，再次开机，观察到是风扇在转从而发出“嗡嗡嗡”的声音，上网查推测可能是内存条没插好，所以我就把内存条拔下来又插上去，并且把主板BIOS的纽扣电池取下来重新放上以恢复出厂设置，然后再重试开机，成功了。

---

开机后很卡，并且没法连接无线网络，故先清理了垃圾软件，并且连上了网线。

清理完软件好多了，但是我想试着装下Windows XP或者重新装一个Windows 7，看能不能更流畅些。查看系统信息后，惊讶地发现，原来这台老电脑是32位，只有两个CPU，内存也只有2G！相应的下载了XP、7和10的Windows，并且尝试安装了7和10，最终是成功安装了7，10在安装时遇到了BIOS版本方面的问题，大概是因为Windows 10和过旧版本的BIOS不兼容的问题吧。在之后我又尝试装Ubuntu，但是卡在了initramfs上，推测是装系统之前需要改一下文件系统，因为Windows支持的是ntfs，而Linux支持的是EXT类型的。

其实折腾一圈，用的还是最初的Windows 7.

---

大学、硕士阶段学了很多知识，但是当我真正面对一个实体电脑，才发现这里面的门道有多深，自己还有很远的路要走。

> 啥是X86？在Windows上时，X86对应32位，X64对应64位；在Linux上时，i386对应32位，AMD64对应64位。
