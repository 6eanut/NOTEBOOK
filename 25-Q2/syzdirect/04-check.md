1. 论文
2. 源码
3. 实验

# syzDirect

基于22年6月的syzkaller实现的

## ChoiceTable

syzkaller：系统调用间的优先级；syzdirect：系统调用对间的优先级

系统调用对：目标系统调用-相关系统调用

优先级：由距离目标代码位置来决定

更新：程序被加入语料库时，更新选择表中系统调用对的距离和优先级

## Corpus

syzkaller：覆盖量作为优先级；syzdirect：距离作为优先级

优先级：距离来决定；具体实现是distanceGroup<距离，数量>，每次chooseProgram时都会计算一遍优先级

更新：是否有新覆盖

## Prog

新信息：Tcall、Rcall、Dist

生成：先选系统调用对，而后基于此来扩展程序

变异：变异操作执行后，需要检验系统调用对是否被破坏
