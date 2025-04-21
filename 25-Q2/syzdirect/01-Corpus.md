# Corpus

```go
type Fuzzer struct {
	// ...
	corpusMu     sync.RWMutex
	corpus       []*prog.Prog
	corpusHashes map[hash.Sig]struct{}
	distanceGroup map[uint32]uint32 // distance -> # program in this distance
	// ...
}
```

Corpus的内容：由多个Prog组成，并且记录了Prog们到目标位置距离的分布；

Corpus的作用：Corpus是syzDirect生成的一组Prog，这组Prog是当前能够覆盖内核的最大范围的一组Prog。

## 1 优先级定义

由距离来决定(chooseProgram)，一个距离对应着相应的优先级，距离越小、次数越多意味着优先级越高。

```go
sumPrios := uint32(0)
prioMap := make(map[uint32]uint32, len(fuzzer.distanceGroup))
for distance, count := range fuzzer.distanceGroup {
	if distance < threeQuarterOfDistance {
		prio := (threeQuarterOfDistance - distance) * 1000 / totalWeight
		sumPrios += prio * count
		prioMap[distance] = prio
	}
}
```

## 2 更新策略

### 2-1 情况一

在poll函数中通过RPC从Manager获取其他Fuzzer发现的程序，对每个程序调用addInputFromAnotherFuzzer函数，进而调用addInputToCorpus函数，将程序加入到Corpus中。

### 2-2 情况二

在loop函数中通过triageInput函数来分析程序，在有新覆盖的情况下，多次运行程序以验证稳定性，然后对程序做最小化处理，然后将程序分发给Manager供其他Fuzzer更新，最后调用addInputToCorpus函数，将程序加入到Corpus中。

所以说只要是有稳定新信号，那就能加入Corpus，只不过距离大的话，优先级会很低。

### 2-3 问题

addInputToCorpus函数会在检查程序是否已经存在前就更新distanceGroup，所以会存在一种情况，就是同一个程序，被count记录了多次，不知道是否是有意设计。

```go
func (fuzzer *Fuzzer) addInputToCorpus(p *prog.Prog, sign signal.Signal, sig hash.Sig) {
    // 加锁保证线程安全
    fuzzer.corpusMu.Lock()
  
    // 更新距离统计（min/max/distanceGroup）
    fuzzer.updateExtremeDist(p.Dist) // 更新distanceGroup
  
    // 检查是否已存在相同哈希的程序
    if _, ok := fuzzer.corpusHashes[sig]; !ok {
        // 添加到corpus切片
        fuzzer.corpus = append(fuzzer.corpus, p)
        // 记录哈希防重复
        fuzzer.corpusHashes[sig] = struct{}{}
    }
  
    // 解锁
    fuzzer.corpusMu.Unlock()
  
    // 合并信号（如果有）
    if !sign.Empty() {
        fuzzer.signalMu.Lock()
        fuzzer.corpusSignal.Merge(sign)
        fuzzer.maxSignal.Merge(sign)
        fuzzer.signalMu.Unlock()
    }
}
```

> 除此之外，还会有语料库的最小化，不过syzdirect没做什么改动。

## 3 选择策略

当需要对一个已有的Prog进行Mutate的时候，需要从Corpus中选择。

选择过程(chooseProgram)：

1. 筛选距离最小的前百分之七十五的距离，否则随机选择；
2. 距离越小权重越大，数量越多权重越大，计算累积优先级；
3. 加权随机选择

```go
func (fuzzer *FuzzerSnapshot) chooseProgram(r *rand.Rand, ct *prog.ChoiceTable) *prog.Prog {
	totalCount := uint32(0)
	distGroupItems := make([]struct {
		dist  uint32
		count uint32
	}, 0)
	for distance, count := range fuzzer.distanceGroup {
		distGroupItems = append(distGroupItems, struct {
			dist  uint32
			count uint32
		}{
			dist:  distance,
			count: count,
		})
		totalCount += count
	}
	sort.Slice(distGroupItems, func(i, j int) bool { return distGroupItems[i].dist < distGroupItems[j].dist })

	// 筛选距离最小的前百分之七十五的距离
	threeQuarterCount := (3 * totalCount / 4)
	threeQuarterOfDistance := uint32(0xffffffff)
	for _, item := range distGroupItems {
		if item.count >= threeQuarterCount {
			threeQuarterOfDistance = item.dist
			break
		} else {
			threeQuarterCount -= item.count
		}
	}

	totalWeight := uint32(0)
	for distance := range fuzzer.distanceGroup {
		if distance < threeQuarterOfDistance {
			totalWeight += threeQuarterOfDistance - distance
		}
	}

	// 否则随机选择
	if totalWeight == 0 {
		randIdx := r.Intn(len(fuzzer.corpus))
		return fuzzer.corpus[randIdx]
	}

	// 距离越小权重越大，数量越多权重越大，计算累积优先级
	sumPrios := uint32(0)
	prioMap := make(map[uint32]uint32, len(fuzzer.distanceGroup))
	for distance, count := range fuzzer.distanceGroup {
		if distance < threeQuarterOfDistance {
			prio := (threeQuarterOfDistance - distance) * 1000 / totalWeight
			sumPrios += prio * count
			prioMap[distance] = prio
		}
	}

	// 加权随机选择
	randVal := uint32(r.Int63n(int64(sumPrios)))
	for _, p := range fuzzer.corpus {
		if p.Dist < threeQuarterOfDistance {
			currPrio := prioMap[p.Dist]
			if currPrio > randVal {
				return p
			}
			randVal -= currPrio
		}
	}
	log.Fatalf("select error ??????")
	return nil
}
```

## 4 关键数据结构

### 4-1 FuzzerSnapshot

chooseProgram函数是FuzzerSnapshot的一个方法，其定义如下：

```go
type FuzzerSnapshot struct {
	corpus        []*prog.Prog
	distanceGroup map[uint32]uint32
}
```

通过snapshot函数可以知道FuzzerSnapshot是获取当前Fuzzer的所有程序和程序到达目标点位距离的信息，Fuzzer.corpus的类型是 `[]*prog.Prog`：

```go
func (fuzzer *Fuzzer) snapshot() FuzzerSnapshot {
	fuzzer.corpusMu.RLock()
	tmpGroup := make(map[uint32]uint32, len(fuzzer.distanceGroup))
	for distance, count := range fuzzer.distanceGroup {
		tmpGroup[distance] = count
	}
	fuzzer.corpusMu.RUnlock()
	return FuzzerSnapshot{fuzzer.corpus, tmpGroup}
}
```

### 4-2 distanceGroup

在chooseProgram函数中观察到有一个distanceGroup，它是FuzzerSnapshot的一个成员，代码如下：

```go
// chooseProgram中用distGroupItems(slice)来表示distanceGroup(map)，为了实现排序并筛选前75%的功能
distGroupItems := make([]struct {
		dist  uint32
		count uint32
		}, 0)
```

* 键：程序距离值，表示程序执行路径与目标代码的接近程度；
* 值：具有该距离的程序数量；
* 作用：统计不同距离的程序分布情况，用于优先级计算。

每当新程序被加入到Corpus时，会调用addInputToCorpus函数，进而调用updateExtremeDist函数，来更新distanceGroup：

```go
func (fuzzer *Fuzzer) updateExtremeDist(dist uint32) {
    if dist != prog.InvalidDist {  // 忽略无效距离，InvalidDist uint32 = 0xFFFFFFFF
        // 更新最小距离
        if fuzzer.minDistance > dist {
            fuzzer.minDistance = dist
        }
        // 更新最大距离
        if fuzzer.maxDistance < dist {
            fuzzer.maxDistance = dist
        }
        // 更新 distanceGroup
        fuzzer.distanceGroup[dist] += 1  // 增加该距离的程序计数
    }
}
```

---

[关键代码](corpus.drawio.png)

```go
// 需要额外关注的差别(syzkaller和syzdirect)：

// loop函数：
HasTcall：确定程序的Tcall和Rcall，引导变异和监控重点
GenerateInGo：相比于Generate，强制包含Tcall，明确关联Tcall和Rcall

// triageInput函数：
progDist：表示当前程序执行路径与目标代码（如漏洞点）的接近程度，值越小表示越接近目标
callDist：特定系统调用（如 ioctl）与目标代码的距离，值越大表示该调用能容忍的偏离范围越大
shouldUpdate：标记当前调用是否需要更新 choiceTable 的选择概率，仅对关键调用（Tcall/Rcall）更新 choiceTable
progHitCounts：map[int]ProgHitItem，记录每个基本块被哪些调用命中
getSignalAndCover：添加返回当前调用的距离

// chooseProgram函数
定义了Corpus中程序(距离)的优先级
```
