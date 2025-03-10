# ChoiceTable

**ChoiceTable的内容**：环境信息、call-to-call的优先级以及当前可用的系统调用集合。

* call-to-call的优先级：对于系统调用对(X,Y)，一个包含了系统调用X的Prog，如果加入系统调用Y，Prog出现新覆盖的可能性，表示为 `runs[X][Y]`。
* 当前可用的系统调用集合：enable->某个系统调用是否被人为指定enable或disable，generatable->`runs[X]!=nil`。

**ChoiceTable的作用**：基于call-to-call的优先级和当前可用的系统调用集合，针对某个给定的系统调用，去加权选择一个系统调用。

## 1 call-to-call的优先级

call-to-call的优先级由静态优先级和动态优先级两部分组成，后者的计算依赖于 `Corpus`，两者计算完毕后相加即可。

### 1-1 静态优先级

静态优先级的计算是基于对系统调用对的参数类型的分析，如果系统调用X和Y都接受相同类型的参数，那么包含它们的Prog更有可能出现新覆盖。

* 原因猜测：相比于无脑组合系统调用，因为参数类型相同而组合系统调用，更有可能触发某些特定的路径。

静态优先级的计算过程：

1. 计算uses：遍历系统调用集合中的每一个系统调用，为每一个系统调用的每一个参数类型和返回值类型赋in和inout权值。
2. 根据uses计算prios：遍历每个类型，对于每个类型中的任何两个系统调用，结合两个系统调用的in和inout权值，计算prios。
3. 处理self-priority：对于系统调用对(X,X)，优先级应该高，但不应过高。
4. 归一化prios：对每一个prios[X][]进行归一化。

> uses的类型是map[string]map[int]weights，weights结构体包含系统调用编号以及in和inout权值，含义是uses[参数类型/返回值类型][系统调用编号]=weights。

### 1-2 动态优先级

动态优先级的计算是基于系统调用对出现在同一个Prog的情况在Corpus中的频率，如果系统调用对出现在同一个Prog的情况在Corpus中很常见，那么就会拥有更高的优先级。

动态优先级的计算过程：

1. 根据Corpus计算prios：遍历Corpus中的每一个Prog，遍历每一个Prog中的每一对系统调用(X,Y)，并统计(X,Y)出现的次数，作为prios。
2. 非线性变换prios：对原值进行开方并乘二，开方是为了压缩大值和保留小值，乘二是为了放大结果，确保变换后的值仍然具有一定的动态范围。
3. 归一化prios：对每一个prios[X][]进行归一化。

## 2 ChoiceTable的更新

当需要读ChoiceTable(生成或变异Prog)，且当Corpus中的程序增多n个时，fuzzer会更新ChoiceTable。

ChoiceTable的更新过程：

1. 确定哪些系统调用可用：结合enable和Corpus。
2. 计算prios：分别计算静态和动态优先级。
3. 计算runs：是prios的累积优先级。

## 3 加权选择

在生成Prog和Prog变异(插入一个系统调用)时，会需要生成系统调用，此时会从ChoiceTable中加权选择系统调用Y。

加权选择的过程：

1. 有5%的概率从ChoiceTable中随机选一个系统调用。
2. 如果没指定系统调用X，就随机指定一个。
3. 根据 `runs[X]`进行加权选择。

> 关于指定系统调用X：
>
> * 如果是生成Prog，那么就是随机指定一个系统调用X；
> * 如果是Prog变异(插入一个系统调用)，那么首先会随机选择一个要插入的位置idx(越靠后被选中的概率越大)，然后从Prog的头到idx之间随机选择一个系统调用X。
