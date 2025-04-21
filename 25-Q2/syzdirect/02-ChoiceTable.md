# ChoiceTable

```go
type ChoiceTable struct {
	target *Target
	runs   [][]int32
	calls  []*Syscall

	GoEnable  bool
	startTime time.Time
	CallPairSelector
}

type CallPairSelector struct {
	hitIndex          uint32
	prioSum           int
	lastHitDataUpdate time.Time
	isUpdated     bool
	callPairInfos []CallPairInfo		// 关键
	infoIdxMap    map[int]map[int]int	// 索引
	mu sync.RWMutex
}

type CallPairInfo struct {
	Tcall int
	Rcall int
	Dists []uint32
	Prio int
}
```

fuzzer.go的main函数为入手点：

1. 通过Corpus和CallPairMap，对includedCalls初始化；
2. 而后构建choicetable，并enablego；
3. 而后关注loop和poll两个函数；

## 1 优先级定义

基于距离的优先级：

* 调用对的优先级由历史距离数据（`Dists []uint32`）计算：

  ```go
  func distance2Prio(distSum uint32, distSize int) int {
      avgDist := float64(distSum) / float64(distSize)
      if avgDist < 1000 {
          return int(1000 * math.Exp(avgDist*(-0.002)))  // 指数衰减
      } else {
          return linearInterpolation(avgDist)  // 线性插值
      }
  }
  ```
* 越小越优先：平均距离越小，优先级越高。

## 2 更新策略

### 2-1 初始化

* 从 `CallPairMap` 和 `RpcCallPairMap` 加载调用对。
* 默认优先级：
  * 有关联调用（`Rcall != -1`）：`prio = 1`。
  * 无关联调用（`Rcall = -1`）：`prio = 0`

### 2-2 动态更新

* 触发条件：程序执行后，`triageInput` 调用 `UpdateCallDistance`。
* 更新逻辑：
  1. 插入新距离到 `Dists` 数组（最多保留 5 个最小距离）。
  2. 重新计算优先级：

```go
prevPrio := info.Prio
newPrio := distance2Prio(calcDistSum(dists), len(dists))
selector.prioSum += (newPrio - prevPrio)
```

## 3 选择策略

`SelectCallPair`：

1. 如果 `prioSum == 0`，随机选择调用对。
2. 否则按优先级加权随机选择：

```go
randVal := r.Intn(selector.prioSum)
for _, info := range selector.callPairInfos {
    if info.Prio > randVal {
        return info.Tcall, info.Rcall
    }
    randVal -= info.Prio
}
```

## 4 关键函数/变量

### 4-1 CallPairMap

含义：Target-Relates Call Pairs，Tcall ID->Rcall IDs

```go
type CallPairMap map[int][]int
```

对CallPairMap的初始化：

```go
type RawCallPair struct {
	Target string
	Relate []string
}

func CallPairFromFile(filename string, target *Target) CallPairMap {			// 从json文件读取Tcall和Rcalls，并转化为CallPairMap格式
	if filename == "" {
		return nil
	}
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Printf("open callfile %v with err: %v\n", filename, err)
		return nil
	}
	var rawCallPairs []RawCallPair							// 先将json转化成[]RawCallPair，Tcall和Rcall都用string来表示
	err = json.Unmarshal(b, &rawCallPairs)
	if err != nil {
		log.Fatalf("call pair unmarshal file %v err: %v\n", filename, err)
	}

	str2Calls := func(call string) []int {
		var res []int
		for _, meta := range target.Syscalls {
			if matchSyscall(meta.Name, call) {
				res = append(res, meta.ID)
			}
		}
		if len(res) == 0 {
			log.Printf("unknown input call:%v\n", call)
		}
		return res
	}

	tmpCallMap := make(map[int]map[int]bool, len(rawCallPairs))			// 将[]RawCallPair转化成tmpCallMap
	for _, rawCallPair := range rawCallPairs {					// Tcall和Rcall由用string表示转化成用ID表示，并用map去重
		tcalls := str2Calls(rawCallPair.Target)
		for _, tcall := range tcalls {
			rcallMap := tmpCallMap[tcall]
			if rcallMap == nil {
				rcallMap = make(map[int]bool, len(rawCallPair.Relate))
				tmpCallMap[tcall] = rcallMap
			}
			for _, rawRCall := range rawCallPair.Relate {
				for _, rc := range str2Calls(rawRCall) {
					rcallMap[rc] = true
				}
			}
		}
	}
	callPairMap := make(CallPairMap, len(tmpCallMap))				// 将tmpCallMap转化成CallPairMap
	for tcall, rcallMap := range tmpCallMap {					// 将map转换成[]，并排序
		keys := make([]int, 0, len(rcallMap))
		for k := range rcallMap {
			keys = append(keys, k)
		}
		sort.Slice(keys, func(i, j int) bool {
			return keys[i] < keys[j]
		})
		callPairMap[tcall] = keys
	}
	return callPairMap
}
```

json格式的文件会在extract_syscall_entry阶段，通过idx(内核版本)和xidx(目标点位)来获取。

### 4-2 includedCalls

含义：Tcall->Rcall->included

```go
var includedCalls map[int]map[int]bool
```

在syz-fuzzer/fuzzer.go的main函数中通过下面代码被初始化：

```go
// manager.Call() 是一个 RPC 调用，它会向 manager 发送 "Manager.Check" 请求，并传递 r.CheckResult 作为参数，包含可用的系统调用列表等。
// manager 端处理这个请求后，会将结果通过第三个参数 &includedCalls 返回。这里的 &includedCalls 是一个指针，所以 manager 可以通过这个指针来修改 includedCalls 的值。
// 这个调用初始化了 includedCalls 这个 map，它会被填充为 manager 返回的数据结构，即一个嵌套的 map map[int]map[int]bool，表示已经包含的调用对。
// 如果调用失败，会通过 log.Fatalf 终止程序。
func main() {
	// ...
	if err := manager.Call("Manager.Check", r.CheckResult, &includedCalls); err != nil {
		log.Fatalf("Manager.Check call failed: %v", err)
	}
	// ...
}
```

在syz-manager/rpc.go的Check函数中定义了如下过程：

```go
// serv.targetEnabledSyscalls由a.EnabledCalls转换而来
func (serv *RPCServer) Check(a *rpctype.CheckArgs, r *map[int]map[int]bool) error {
	// ...
	includedCalls := serv.mgr.machineChecked(a, serv.targetEnabledSyscalls)
	// ...
	*r = includedCalls
    	return nil
}
```

在syz-manager/manager.go的machineChecked函数中定义了如下过程：

```go
func (mgr *Manager) machineChecked(a *rpctype.CheckArgs, enabledSyscalls map[*prog.Syscall]bool) map[int]map[int]bool {
	mgr.mu.Lock()
	defer mgr.mu.Unlock()
	mgr.checkResult = a
	mgr.targetEnabledSyscalls = enabledSyscalls
	mgr.target.UpdateGlobs(a.GlobFiles)
	tmp := mgr.loadCorpus()
	mgr.firstConnect = time.Now()
	return tmp
}
```

在syz-manager/manager.go的loadCorpus函数中定义了如下过程：

```go
// loadCorpus函数的作用是加载持久化语料库，并利用loadProg将符合条件的程序加入候选队列
// syzdirect在该过程中获得Tcalls，并指导loadProg，筛选出包含Tcall的程序，而后初始化includedCalls
func (mgr *Manager) loadCorpus() map[int]map[int]bool {
	// ...
	rawTcalls := mgr.callPairMap.GetRawTargetCalls(mgr.target)			// 获得Tcalls
	for key, rec := range mgr.corpusDB.Records {					// 加载持久化语料库
		if !mgr.loadProg(rec.Val, minimized, smashed, rawTcalls) {		// 利用loadProg将符合条件的程序加入候选队列，Tcalls指导loadProg的过程
			mgr.corpusDB.Delete(key)
			broken++
		}
	}
	// 初始化includedCalls的过程
	// ...
}
```

获得Tcalls？指导loadProg的过程？初始化includedCalls的过程？将在这里说明：

```go
// 获得Tcalls
func (cpMap CallPairMap) GetRawTargetCalls(target *Target) map[int]bool {		// 将CallPairMap类型的cpMap转化成map[int]bool类型的rawTcalls，即只保留Tcall
	rawTcalls := make(map[int]bool, len(cpMap))
	for tcall := range cpMap {
		callName := target.Syscalls[tcall].Name					// 处理带有 "_rf1" 或 "$tmp_rf1" 后缀的变异调用
		if strings.HasSuffix(callName, "_rf1") {
			oriName := callName[:len(callName)-4]				// 将变异调用还原为原始调用
			if strings.HasSuffix(callName, "$tmp_rf1") {
				oriName = callName[:len(callName)-8]
			}
			rawTcalls[target.SyscallMap[oriName].ID] = true
		} else {
			rawTcalls[tcall] = true
		}
	}
	return rawTcalls
}

// 指导loadProg的过程
func (mgr *Manager) loadProg(data []byte, minimized, smashed bool, rawTcalls map[int]bool) bool {
	// ...
	hasTarget := false								// hasTarget记录prog是否包含Tcall
	for _, call := range p.Calls {							// 遍历prog中的每个call，检查是不是Tcall
		if rawTcalls[call.Meta.ID] {
			hasTarget = true
			break
		}
	}
	if !hasTarget {									// 如果prog不含Tcall，那么不会被加入到candidates中
		return false
	}
	mgr.candidates = append(mgr.candidates, rpctype.Candidate{			// 如果prog包含Tcall，那么会加入到candidates中
		Prog:      data,
		Minimized: minimized,
		Smashed:   smashed,
	})
	return true
}

// 初始化includedCalls的过程
func (mgr *Manager) loadCorpus() map[int]map[int]bool {
	// ...
	includedCalls := make(map[int]map[int]bool)
	for i := range mgr.candidates {								// 遍历candidates中的每个prog
		p, err := mgr.target.Deserialize(mgr.candidates[i].Prog, prog.NonStrict)
		if err != nil {
			log.Fatalf("should success")
		}
		for i := len(p.Calls) - 1; i >= 0; i-- {					// 从prog中逆序遍历call，找到所有Tcalls，每个Tcall对应的是Rcalls
			if rcalls, ok := mgr.serv.callPairMap[p.Calls[i].Meta.ID]; ok {
				tcallId, rcallId := p.Calls[i].Meta.ID, -1
				for j := 0; j < i && rcallId != -1; j++ {			// 从prog中顺序遍历call直到Tcall，看是否在Rcalls中
					for _, rcall := range rcalls {
						if rcall == p.Calls[j].Meta.ID {
							rcallId = rcall				// 如果找到了Tcall对应的Rcall，则记录，否则Rcall为-1
							break
						}
					}
				}
				if includedCalls[tcallId] == nil {				// 只要找到Tcall，无论找没找到Rcall，都写includedCalls
					includedCalls[tcallId] = make(map[int]bool)
				}
				includedCalls[tcallId][rcallId] = true
				break
			}
		}
	}
	log.Logf(0, "[syzgo] includedCalls: %v", includedCalls)
	// ...
	return includedCalls
}
```

### 4-3 BuildChoiceTable

在syz-fuzzer/fuzzer.go的main函数中通过下面代码被初始化：

```go
type Fuzzer struct {
	// ...
	choiceTable       *prog.ChoiceTable
	// ...
}

type ChoiceTable struct {
	target *Target
	runs   [][]int32
	calls  []*Syscall

	GoEnable  bool
	startTime time.Time
	CallPairSelector												// 后面会详解
}

func main() {
	// ...
	fuzzer.choiceTable = target.BuildChoiceTable(fuzzer.corpus, calls)						// corpus最初为空，calls为系统支持的calls，BuildChoiceTable的过程和syzkaller相同
	fuzzer.choiceTable.EnableGo(r.CallPairMap, r.RpcCallPairMap, fuzzer.corpus, fuzzer.startTime, r.HitIndex)	// 后面详解
	// ...

```

### 4-4 EnableGo

在prog/direct.go中定义了EnableGo的过程：

```go
，// 依照CallPairMap(静态)和rpcCallPairMap(动态，包含dist)来对ChoiceTable的CallPairSelector做初始化
func (ct *ChoiceTable) EnableGo(cpMap CallPairMap, rpcCPMap RpcCallPairMap, corpus []*Prog, startTime time.Time, hitIndex uint32) {
	cpInfos := make([]CallPairInfo, 0, len(cpMap)*3)						// 初始化 CallPairInfo 列表和infoIdxMap索引映射
	infoIdxMap := make(map[int]map[int]int, len(cpMap))
	allPrio := 0
	for tcall, rcalls := range cpMap {								// 遍历CallPairMap的所有调用对（Tcall → []Rcall）
		if !ct.Enabled(tcall) {
			continue
		}
		rpcRcallMap, ok1 := rpcCPMap[tcall]
		tmp := append(rcalls, -1)								// 处理每个Rcall（包括无关联调用 Rcall=-1）
		rIdxMap := make(map[int]int)
		for _, rcall := range tmp {
			if rcall != -1 && !ct.Enabled(rcall) {
				continue
			}
			hasAdd := false
			if ok1 {
				if dists2, ok2 := rpcRcallMap[rcall]; ok2 {				// 从 rpcCallPairMap 加载历史距离数据（如有）
					prio := distance2Prio(calcDistSum(dists2), len(dists2))
					cpInfos = append(cpInfos, CallPairInfo{
						Tcall: tcall,
						Rcall: rcall,
						Prio:  prio,
						Dists: dists2,
					})
					allPrio += prio
					hasAdd = true
				}
			}
			if !hasAdd {									// 默认优先级（无历史数据时），有关联prio=1，无关联prio=0
				prio := 1
				if rcall == -1 {
					prio = 0
				}
				cpInfos = append(cpInfos, CallPairInfo{
					Tcall: tcall,
					Rcall: rcall,
					Prio:  prio,
					Dists: make([]uint32, 0, 5),
				})
				allPrio += prio
			}
			rIdxMap[rcall] = len(cpInfos) - 1
		}
		// targetCalls = append(targetCalls, tcall)
		infoIdxMap[tcall] = rIdxMap
	}
	if len(cpInfos) == 0 {
		panic("all target calls are disabled")
	}

	ct.GoEnable = true										// 更新ChoiceTable中的CallPairSelector
	ct.startTime = startTime
	ct.CallPairSelector.hitIndex = hitIndex
	ct.CallPairSelector.callPairInfos = cpInfos
	ct.CallPairSelector.prioSum = allPrio
	ct.CallPairSelector.infoIdxMap = infoIdxMap
}
```

### 4-5 CallPairSelector

BuildChoiceTable将ChoiceTable的普通内容填写(syzakller)，EnableGo将ChoiceTable的系统调用对内容填写(syzdirect)，即CallPairSelector，定义在prog/direct.go：

```go
type CallPairSelector struct {
	hitIndex          uint32
	prioSum           int
	lastHitDataUpdate time.Time
	isUpdated     bool
	callPairInfos []CallPairInfo		// 关键
	infoIdxMap    map[int]map[int]int	// 索引
	mu sync.RWMutex
}

type CallPairInfo struct {
	Tcall int
	Rcall int
	Dists []uint32
	Prio int
}

// CallPairSelector负责存储和管理调用对及其距离信息，根据距离信息计算每个调用对的优先级，根据优先级随机选择调用对；
// EnableGo方法从cpMap和rpcCPMap构建调用对信息，计算初始优先级；
// UpdateCallDistance根据程序中的Tcall和Rcall找到对应的CallPairInfo，将新距离插入到Dists数组中，重新计算优先级
// SelectorCallPair根据优先级随机选择调用对
// distance2Prio实现距离到优先级的转换
```

下面将分别记录对于CallPairSelector的一些方法：

#### 4-5-1 UpdateCallDistance

```go
// 每个Prog都包含调用对信息和距离
type Prog struct {
	Target   *Target
	Calls    []*Call
	Comments []string
	ProgExtra
}

type ProgExtra struct {
	Dist  uint32
	Tcall *Call
	Rcall *Call
}

// loop->triageInput->UpdateCallDistance
// triageInput会多次执行程序以确定信号的稳定性，进行最小化，并将程序加入到语料库中
// 如果该程序中Tcall和Rcall的距离有效，则会调用UpdateCallDistance来对距离和优先级做更新
func (selector *CallPairSelector) UpdateCallDistance(p *Prog, dist uint32) {
	if dist == InvalidDist {
		return
	}
	selector.mu.Lock()
	defer selector.mu.Unlock()
	tcallId := p.Tcall.Meta.ID								// 获取程序的Tcall和Rcall
	rcallId := -1
	if p.Rcall != nil {
		rcallId = p.Rcall.Meta.ID
	}
	infoIdx := selector.infoIdxMap[tcallId][rcallId]					// 获取Tcall&Rcall的系统调用对信息CallPairInfo
	info := &selector.callPairInfos[infoIdx]
	dists := info.Dists
	idx, shouldRet := locateIndex(dists, dist)						// 定位是否要将新距离插入到距离数组中，以及确定要插入的位置
	if shouldRet {
		return
	}
	prevDistSum := calcDistSum(dists)
	if idx == len(dists) {									// 如果新距离应该插入到数组末尾
		dists = append(dists, dist)
	} else {
		if len(dists) >= 5 {
			dists = dists[:4]							// 如果数组已满(长度>=5)，截断只保留前4个
		}
		if idx == 0 {									// 如果新距离应该插入到数组开头
			right := len(dists) - 1
			for right >= 0 && 2*dist < dists[right] {				// 数组中的最大元组必须小于等于最小元素的两倍，所以需要筛选掉大于新距离两倍的距离
				right--
			}
			if right >= 0 {
				dists = append([]uint32{dist}, dists[:right+1]...)
			} else {
				dists = []uint32{dist}
			}
		} else {									// 对于中间位置的插入
			tmp := append([]uint32{dist}, dists[idx:]...)
			dists = append(dists[:idx], tmp...)
		}
	}
	currDistSum := calcDistSum(dists)
	info.Dists = dists
	if prevDistSum != currDistSum {								// CallPairInfo的距离数组和变化时，更新优先级
		selector.prioSum = selector.prioSum - info.Prio
		info.Prio = distance2Prio(currDistSum, len(info.Dists))
		selector.prioSum += info.Prio
		selector.isUpdated = true
	}
}

// 在dists中找到dist的位置
func locateIndex(dists []uint32, dist uint32) (int, bool) {
	idx := len(dists) - 1
	for idx >= 0 {										// 找到第一个不大于新距离的元素，停止查找
		if dists[idx] > dist {
			idx -= 1
		} else {
			break
		}
	}
	idx += 1
	if idx >= 5 || (len(dists) > 0 && CallPairLimitMulti*dists[0] < dist) {			// 数组已满且新距离较大 或 新距离是原最小距离的两倍以上，则不该加入新距离
		return idx, true
	}
	return idx, false
}

// 当CallPairInfo的Dists和变化时，更新优先级
func distance2Prio(distSum uint32, distSize int) int {
	var prio int
	dist := float64(distSum) / float64(distSize)						// 计算平均距离，根据平均距离范围采取不同的优先级计算策略
	if dist < 1000 {									// 小于1000，指数衰减，距离越小，优先级越高
		prio = int(1000 * math.Exp(dist*(-0.002)))
	} else {
		left, right := 0.0, 0.0
		switch int(dist / 1000) {							// 大于1000，将距离按每1000为一个区间划分
		case 1:
			left, right = 135, 48
		case 2:
			left, right = 48, 16
		case 3:
			left, right = 16, 8
		case 4:
			left, right = 8, 4
		case 5:
			left, right = 4, 2
		}
		if left == right {								// 大于6000，prio为1
			prio = 1
		} else {
			prio = int(left - (left-right)*(float64(int(dist)%1000))/1000.0)	// 线性插值计算
		}
	}
	return prio
}
```

#### 4-5-2 SelectCallPair

```go
// loop->GenerateInGo->SelectCallPair
// loop->Mutate->FixExtraCalls->SelectCallPair
// 生成或变异程序，都会需要选择系统调用对
func (selector *CallPairSelector) SelectCallPair(r *rand.Rand) (int, int) {
	selector.mu.RLock()
	defer selector.mu.RUnlock()
	if selector.prioSum == 0 {									// 优先级为0，随机选择，系统调用之间均无关联
		idx := r.Intn(len(selector.callPairInfos))
		info := &selector.callPairInfos[idx]
		return info.Tcall, info.Rcall
	}
	randVal := r.Intn(selector.prioSum)
	for i := range selector.callPairInfos {								// 累积优先级加权选择
		info := &selector.callPairInfos[i]
		if info.Prio > randVal {
			return info.Tcall, info.Rcall
		}
		randVal -= info.Prio
	}
	log.Fatalf("what ??????")
	return -1, -1
}

// 下面分别说明生成和变异两种情况

// 生成
// 在loop中，当语料库为空 或 完成一个周期时，会调用GenerateInGo生成测试程序
func (target *Target) GenerateInGo(rs rand.Source, ncalls int, ct *ChoiceTable) *Prog {
	if !ct.GoEnable {
		return target.Generate(rs, ncalls, ct)
	}
	tcallId, rcallId := ct.SelectCallPair(rand.New(rs))						// 选择一个调用对
	// log.Printf("tcall id: %v, rcall id: %v\n", tcallId, rcallId)
	return target.generateHelper(ct, rs, ncalls, tcallId, rcallId)					// generateHelper会根据调用对生成程序
}

func (target *Target) generateHelper(ct *ChoiceTable, rs rand.Source, ncalls, tcallId, rcallId int) *Prog {
	var rcall *Call
	s := newState(target, ct, nil)
	r := newRand(target, rs)
	p := &Prog{											// 初始化Prog p
		Target: target,
		ProgExtra: ProgExtra{
			Dist: InvalidDist,
		},
	}

	if rcallId != -1 {										// 处理Rcall，后续会说明generateParticularCall
		rcalls := r.generateParticularCall(s, target.Syscalls[rcallId])
		for _, c := range rcalls {
			s.analyze(c)
			p.Calls = append(p.Calls, c)
		}
		rcall = rcalls[len(rcalls)-1]
	}

	for len(p.Calls) < ncalls-1 {									// 填充中间调用，后续会说明generateCall
		calls := r.generateCall(s, p, len(p.Calls))
		for _, c := range calls {
			s.analyze(c)
			p.Calls = append(p.Calls, c)
		}
	}

	r.rcall = rcall
	targetCalls := r.generateParticularCall(s, r.target.Syscalls[tcallId])				// 处理Tcall
	p.Rcall = rcall
	p.Tcall = targetCalls[len(targetCalls)-1]

	rmIdx := len(p.Calls) - 1
	if rmIdx < 0 {
		rmIdx = 0
	}
	p.Calls = append(p.Calls, targetCalls...)
	for len(p.Calls) > ncalls {									// 调整程序长度
		isSucc := p.RemoveCall(rmIdx)
		if !isSucc && rmIdx == 0 {
			rmIdx = 1
		} else if rmIdx > 0 {
			rmIdx--
		}
	}
	return p
}

// 变异
// 当程序需要变异时，变异之后的程序的调用对信息可能被破坏，这时需要对程序p修复，故用到了SelectCallPair
func (p *Prog) Mutate(rs rand.Source, ncalls int, ct *ChoiceTable, corpus []*Prog) {
	r := newRand(p.Target, rs)									// 初始化
	if ncalls < len(p.Calls) {
		ncalls = len(p.Calls)
	}
	ctx := &mutator{
		p:      p,
		r:      r,
		ncalls: ncalls,
		ct:     ct,
		corpus: corpus,
	}
	for stop, ok := false, false; !stop; stop = ok && len(p.Calls) != 0 && r.oneOf(3) {		// 执行变异操作
		switch {
		case r.oneOf(5):
			// Not all calls have anything squashable,
			// so this has lower priority in reality.
			ok = ctx.squashAny()								// 尝试压缩复杂指针
		case r.nOutOf(1, 100):
			ok = ctx.splice()								// 程序拼接
		case r.nOutOf(20, 31):
			x := float64(time.Since(ct.startTime) / time.Minute)				// 随时间变化的混合变异策略
			y0 := math.Pow(20, x/(-50)) / 2
			y1 := y0 + 0.5
			y2 := -y0 + 1.0
			if y1 > r.Float64() {
				rcallIdx := getCallIndexByPtr(ctx.p.Rcall, ctx.p.Calls)
				if rcallIdx != -1 && r.oneOf(2) {
					ok = ctx.mutateArg(rcallIdx)
				} else {
					ok = ctx.mutateArg(getCallIndexByPtr(ctx.p.Tcall, ctx.p.Calls))
				}
			}
			if y2 > r.Float64() {
				ok = ok || ctx.insertCall()
			}
		case r.nOutOf(10, 11):
			ok = ctx.mutateArg(-1)								// 随机变异参数
		default:
			ok = ctx.removeCall()								// 移除随机调用
		}
	}
	if p.Tcall == nil {
		p.Target.FixExtraCalls(p, r.Rand, ct, RecommendedCalls, nil)				// 如果变异破坏了目标调用对关系，调用FixExtraCalls修复，该函数和generateHelper类似
	}
	p.sanitizeFix()											// 确保程序结构有效
	p.debugValidate()										// 验证
	if got := len(p.Calls); got < 1 || got > ncalls {
		panic(fmt.Sprintf("bad number of calls after mutation: %v, want [1, %v]", got, ncalls))
	}
}
```

### 4-6 HasTcall

当语料库不为空时，HasTcall会遍历语料库中的每一个程序，然后表注Tcall和Rcall。

Tcall是从后向前遍历call，遇到的第一个Tcall；Rcall是从前向后遍历call，遇到的第一个Rcall。

```go
func (p *Prog) HasTcall(ct *ChoiceTable) bool {
	if p.Tcall != nil {
		return true
	}
	for i := len(p.Calls) - 1; i >= 0; i-- {
		if rcallMap, ok := ct.infoIdxMap[p.Calls[i].Meta.ID]; ok {
			p.Tcall = p.Calls[i]
			p.Rcall = nil
			for j := 0; j < i; j++ {
				if _, ok = rcallMap[p.Calls[j].Meta.ID]; ok {
					p.Rcall = p.Calls[j]
					break
				}
			}
			return true
		}
	}
	return false
}
```

### 4-7 generateCandidateInputInGo

根据includedCalls来生成candidates，只包含Tcall或Rcall+Tcall。

```go
func (fuzzer *Fuzzer) generateCandidateInputInGo(includedCalls map[int]map[int]bool) {
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	progs := fuzzer.target.MultiGenerateInGo(rnd, fuzzer.choiceTable, includedCalls)
	for _, p := range progs {
		fuzzer.workQueue.enqueue(&WorkCandidate{
			p:     p,
			flags: ProgCandidate,
		})
	}
}

func (target *Target) MultiGenerateInGo(rs rand.Source, ct *ChoiceTable, includedCalls map[int]map[int]bool) []*Prog {
	progs := make([]*Prog, 0, len(ct.callPairInfos)*CallPairInitNum)
	for i := 0; i < CallPairInitNum; i++ {
		for j := 0; j < len(ct.callPairInfos); j++ {
			inf := &ct.callPairInfos[j]
			if len(ct.infoIdxMap[inf.Tcall]) > 1 && inf.Rcall == -1 {
				continue
			}
			if rcallMap, ok := includedCalls[inf.Tcall]; ok && (inf.Rcall == -1 || rcallMap[inf.Rcall]) {
				continue
			}
			ncalls := 2
			if inf.Rcall == -1 {
				ncalls = 1
			}
			progs = append(progs, target.generateHelper(ct, rs, ncalls, inf.Tcall, inf.Rcall))
		}
	}
	return progs
}
```

### 4-8 loop

负责fuzzing，不断从工作队列中获取任务并执行。

1. 从工作队列取出一个任务（Triage 或 Candidate）→ 处理它。
2. 如果没有任务：
   - 生成全新程序 → 执行（`GenerateInGo` + `executeAndCollide`）。
   - 或从语料库选择程序 → 变异 → 执行（`chooseProgram` + `Mutate` + `executeAndCollide`）。
3. 执行过程中若发现新信号 → 生成新的 `WorkTriage` 任务加入队列。

```go
func (proc *Proc) loop() {
	generatePeriod := 100									// 每 100 次循环生成一个新程序
	if proc.fuzzer.config.Flags&ipc.FlagSignal == 0 {
		// If we don't have real coverage signal, generate programs more frequently
		// because fallback signal is weak.
		generatePeriod = 2
	}
	for i := 0; ; i++ {
		item := proc.fuzzer.workQueue.dequeue()
		if item != nil {
			switch item := item.(type) {
			case *WorkTriage:							// 处理新发现的信号
				proc.triageInput(item)
			case *WorkCandidate:
				item.p.HasTcall(proc.fuzzer.choiceTable)
				proc.execute(proc.execOpts, item.p, item.flags, StatCandidate)	// 处理候选程序
			case *WorkSmash:
				proc.smashInput(item)						// 暴力变异
			default:
				log.Fatalf("unknown work type: %#v", item)
			}
			continue
		}

		ct := proc.fuzzer.choiceTable
		fuzzerSnapshot := proc.fuzzer.snapshot()
		if len(fuzzerSnapshot.corpus) == 0 || i%generatePeriod == 0 {			// 生成新程序
			// Generate a new prog.
			p := proc.fuzzer.target.GenerateInGo(proc.rnd, prog.RecommendedCalls, ct)
			log.Logf(1, "#%v: generated", proc.pid)
			proc.executeAndCollide(proc.execOpts, p, ProgNormal, StatGenerate)
		} else {									// 对程序进行变异
			// Mutate an existing prog.
			p := fuzzerSnapshot.chooseProgram(proc.rnd, proc.fuzzer.choiceTable).Clone()
			p.Mutate(proc.rnd, prog.RecommendedCalls, ct, fuzzerSnapshot.corpus)
			log.Logf(1, "#%v: mutated", proc.pid)
			proc.executeAndCollide(proc.execOpts, p, ProgNormal, StatFuzz)
		}
	}
}
```

### 4-9 poll

负责和Manager通信

```go
func (fuzzer *Fuzzer) poll(needCandidates bool, stats map[string]uint64) bool {
	tqs, sqs, cqs := fuzzer.workQueue.getSize()
	a := &rpctype.PollArgs{
		Name:               fuzzer.name,
		NeedCandidates:     needCandidates,
		MaxSignal:          fuzzer.grabNewSignal().Serialize(),
		Stats:              stats,
		TriageQueueSize:    tqs,
		SmashQueueSize:     sqs,
		CandidateQueueSize: cqs,
	}
	r := &rpctype.PollRes{}
	if err := fuzzer.manager.Call("Manager.Poll", a, r); err != nil {
		log.Fatalf("Manager.Poll call failed: %v", err)
	}
	maxSignal := r.MaxSignal.Deserialize()
	log.Logf(1, "poll: candidates=%v inputs=%v signal=%v",
		len(r.Candidates), len(r.NewInputs), maxSignal.Len())
	fuzzer.addMaxSignal(maxSignal)
	for _, inp := range r.NewInputs {			// 获取inputs
		fuzzer.addInputFromAnotherFuzzer(inp)
	}
	for _, candidate := range r.Candidates {		// 获取candidate
		fuzzer.addCandidateInput(candidate)
	}
	if needCandidates && len(r.Candidates) == 0 && atomic.LoadUint32(&fuzzer.triagedCandidates) == 0 {
		atomic.StoreUint32(&fuzzer.triagedCandidates, 1)
	}
	return len(r.NewInputs) != 0 || len(r.Candidates) != 0 || maxSignal.Len() != 0
}
```


### 4-10 Hit

推测可能是为了离线分析，因为每个五分钟，会把hit信息写到json文件里面

#### 4-10-1 ProgHitCountItem

Exec->parseOutput->MergeHitCount：执行测试程序，收集执行结果，并记录

```go
type ProgHitCountItem struct {
	Count   uint32
	CallIds []int
}

type ProgHitCounts map[uint32]ProgHitCountItem				# 索引是地址，值是触发该地址的次数和相应的系统调用；和程序一一对应

func (progHitCount ProgHitCounts) MergeHitCount(hitArr []uint32, callId int) {
	if len(hitArr)%2 != 0 {
		log.Fatalf("hit array not dual %v", hitArr)
	}
	for i := 0; i < len(hitArr); i += 2 {
		hitItem := progHitCount[hitArr[i]]			# 第奇数个数记录的是地址
		hitItem.Count += hitArr[i+1]				# 第偶数个数记录的触发该地址的次数
		hitItem.CallIds = append(hitItem.CallIds, callId)	# 把触发该地址的系统调用加进去
		progHitCount[hitArr[i]] = hitItem			# 更新
	}
}
```

#### 4-10-2 HitLogItem

```go
type HitLogItem struct {
	Count        uint32
	HitCalls     []int
	FirstHitTime time.Duration
	Progs        []string
}

type GlobalHitLog map[uint32]HitLogItem						# 索引是地址，值是触发该地址的程序、第一次触发的时间、触发的次数以及相应的系统调用

func (item *HitLogItem) AddCallIds(callids []int) {				# 为某个地址，更新触发该地址的系统调用
	for _, newCallid := range callids {
		hasRecord := false
		for _, oldCallid := range item.HitCalls {
			if oldCallid == newCallid {
				hasRecord = true
				break
			}
		}
		if !hasRecord {
			item.HitCalls = append(item.HitCalls, newCallid)
		}
	}
}

func (hitLog GlobalHitLog) CustomMarshal(target *Target) ([]byte, error) {	# 将callid转化成string
	outMap := make(map[uint32]struct {
		Count        uint32
		HitCalls     []string
		FirstHitTime time.Duration
		Progs        []string
	}, len(hitLog))
	for key, val := range hitLog {
		hitCalls := make([]string, 0, len(val.HitCalls))
		for _, callid := range val.HitCalls {
			callName := "extra"
			if callid > 0 && callid < len(target.Syscalls) {
				callName = target.Syscalls[callid].Name
			}
			hitCalls = append(hitCalls, callName)
		}
		outMap[key] = struct {
			Count        uint32
			HitCalls     []string
			FirstHitTime time.Duration
			Progs        []string
		}{
			Count:        val.Count,
			Progs:        val.Progs,
			HitCalls:     hitCalls,
			FirstHitTime: val.FirstHitTime,
		}
	}
	return json.Marshal(outMap)
}
```

### 4-11 CallMapItem

GenCallRelationData是核心函数：先通过 `getSimpleResources()`分析每个系统调用的输入/输出资源，特别处理文件操作类调用；然后构建各种映射，生成target2relate2.json文件，用于生成CallPairMap

```go
type CallMapItem struct {
	Module      string
	FullVersion []string
	SimpVersion []string
	TrimVersion []string
}

func IsFileGenCallName(name string) bool {
	return strings.HasPrefix(name, "mk") || strings.HasPrefix(name, "open") || name == "creat" ||
		(strings.Contains(name, "mount") && !strings.Contains(name, "umount") && name != "move_mount")
}

func (target *Target) GenCallRelationData() (map[int][]int, map[int][]int) {
	// full version
	// fullCallCtorMap := target.GenTarget2Relate()
	// fullCallUserMap := target.GenRelate2Context()

	// only consider simplest input and output resource
	call2OutRescs := make([][]*ResourceDesc, len(target.Syscalls))
	inpResc2Calls := make(map[*ResourceDesc][]int, len(target.Resources))
	fileGenCalls := make(map[int]bool, 0)
	fileUseCalls := make([]int, 0)

	resc2FullUsers := make(map[*ResourceDesc][]int, len(target.Resources))
	resc2MatchUsers := make(map[*ResourceDesc][]int, len(target.Resources))

	callFullUserMap := make(map[int][]int, len(target.Syscalls))
	callMatchUserMap := make(map[int][]int, len(target.Syscalls))
	callCtorMap := make(map[int][]int, len(target.Syscalls))

	for metaId, meta := range target.Syscalls {
		inpRescs, outRescs, hasFileInput := getSimpleResources(meta)
		if len(outRescs) > 0 {
			call2OutRescs[metaId] = outRescs
		}
		if meta.Name == "sendfile64" {
			fmt.Printf("break here")
		}
		if hasFileInput {
			if IsFileGenCallName(meta.Name) {
				fileGenCalls[metaId] = true
			} else {
				fileUseCalls = append(fileUseCalls, metaId)
			}
		}
		for _, inpResc := range inpRescs {
			inpResc2Calls[inpResc] = append(inpResc2Calls[inpResc], metaId)
		}
	}

	// for call := range fileGenCalls {
	// 	fmt.Printf("file gen call: %v\n", target.Syscalls[call].Name)
	// }
	// for _, call := range fileUseCalls {
	// 	fmt.Printf("file use call: %v\n", target.Syscalls[call].Name)
	// }

	matchStat := 0
	unmatchStat := 0
	for _, res := range target.Resources {
		userMap := make(map[int]int)
		matchUserNum := 0
		for inpResc, calls := range inpResc2Calls {
			if isCompatibleResourceImpl(inpResc.Kind, res.Kind, true) {
				level := 1
				if len(inpResc.Kind) == len(res.Kind) {
					level = 2
					matchUserNum += len(calls)
				}
				for _, call := range calls {
					userMap[call] = level
				}

			}
		}
		if len(userMap) > 0 {
			fullUsers := make([]int, 0, len(userMap))
			var matchUsers []int
			if matchUserNum > 0 {
				matchUsers = make([]int, 0, matchUserNum)
			}
			for call, level := range userMap {
				fullUsers = append(fullUsers, call)
				if level == 2 {
					matchUsers = append(matchUsers, call)
				}
			}
			if matchUserNum == 0 {
				matchUsers = fullUsers
				unmatchStat += 1
				// fmt.Printf("unmatch resc: %v\n", res.Kind)
			} else {
				matchStat += 1
			}
			resc2FullUsers[res] = fullUsers
			resc2MatchUsers[res] = matchUsers
		}
	}

	// fmt.Printf("match: %v, unmatch: %v\n", matchStat, unmatchStat)

	for srcCallId, outRescs := range call2OutRescs {
		allUserMap := make(map[int]int)
		matchUserNum := 0
		if target.Syscalls[srcCallId].Name == "sendfile64" {
			fmt.Printf("break here")
		}

		for _, outResc := range outRescs {
			for _, call := range resc2FullUsers[outResc] {
				allUserMap[call] = 1
			}
			matchUsers := resc2MatchUsers[outResc]
			matchUserNum += len(matchUsers)
			for _, call := range matchUsers {
				allUserMap[call] = 2
			}
		}
		if fileGenCalls[srcCallId] {
			for _, call := range fileUseCalls {
				allUserMap[call] = 2
			}
		}

		if len(allUserMap) > 0 {
			allFullUsers := make([]int, 0, len(allUserMap))
			var allMatchUsers []int
			if matchUserNum > 0 {
				allMatchUsers = make([]int, 0, matchUserNum)
			}
			for call, level := range allUserMap {
				allFullUsers = append(allFullUsers, call)
				if level == 2 {
					allMatchUsers = append(allMatchUsers, call)
				}
			}
			if matchUserNum == 0 {
				allMatchUsers = allFullUsers
			}
			callFullUserMap[srcCallId] = allFullUsers
			callMatchUserMap[srcCallId] = allMatchUsers
		}
	}

	for srcCall, userCalls := range callFullUserMap {
		for _, user := range userCalls {
			callCtorMap[user] = append(callCtorMap[user], srcCall)
		}
	}

	limitCallUserMap := target.limitCallScope(callMatchUserMap)
	limitCallCtorMap := target.limitCallScope(callCtorMap)

	target.outCallMap(callCtorMap, limitCallCtorMap, "target2relate2.json")

	return limitCallCtorMap, limitCallUserMap
}

func (target *Target) limitCallScope(callMap map[int][]int) map[int][]int {
	litmitCallMap := make(map[int][]int, len(callMap))

	moduleParser := make(map[string][]string)
	getNameByLevel := func(module string, level int) string {
		seqs, ok := moduleParser[module]
		if !ok {
			seqs = make([]string, 0)
			curr := module
			for {
				seqs = append(seqs, module)
				idx := strings.LastIndex(curr, "_")
				if idx == -1 {
					break
				}
				curr = curr[:idx]
			}
			moduleParser[module] = seqs
		}
		if level >= len(seqs) {
			return ""
		} else {
			return seqs[level]
		}
	}

	for from, toCalls := range callMap {
		fromMeta := target.Syscalls[from]
		limitToCalls := make([]int, 0, len(toCalls)/2)
		// log.Printf("src module: %v", fromMeta.Module)
		maxIterCount := strings.Count(fromMeta.Module, "_") + 2
		for level := 0; level < maxIterCount; level++ {
			for _, toCall := range toCalls {
				superModule := getNameByLevel(fromMeta.Module, level)
				if superModule == "" {
					superModule = "sys"
				}
				if superModule == target.Syscalls[toCall].Module {
					limitToCalls = append(limitToCalls, toCall)
				}
				// } else {
				// 	log.Printf("un match module: %v", target.Syscalls[call].Module)
				// }
			}
			if len(limitToCalls) > 0 {
				break
			}
		}
		if len(limitToCalls) == 0 {
			limitToCalls = toCalls
		}
		litmitCallMap[from] = limitToCalls
		// else {
		// log.Printf("call %v, src module: %v, all related calls are banned", fromMeta.Name, fromMeta.Module)
		// }
	}
	return litmitCallMap
}

func (target *Target) outCallMap(simpCallMap, sameModuleMap map[int][]int, outFileName string) {
	allCallMap := make(map[string]CallMapItem, len(simpCallMap))

	// for check: require simpCallMap is subset of fullCallMap
	// for fromCall, toCalls := range simpCallMap {
	// 	if fullUserCalls, ok := fullCallMap[fromCall]; ok {
	// 		fullCallMap := make(map[int]bool)
	// 		for _, call := range fullUserCalls {
	// 			fullCallMap[call] = true
	// 		}
	// 		for _, call := range toCalls {
	// 			if !fullCallMap[call] {
	// 				log.Printf("%v: tcall:%v, rcall: %v", outFileName, fromCall, call)
	// 			}
	// 		}
	// 	} else {
	// 		log.Printf("%v: %v not in", outFileName, fromCall)
	// 	}
	// }
	// start to combine data
	callIds2Names := func(raws []int) []string {
		if len(raws) == 0 {
			return nil
		}
		sort.Ints(raws)
		res := make([]string, 0, len(raws))
		for _, raw := range raws {
			res = append(res, target.Syscalls[raw].Name)
		}
		return res
	}

	for fromCall := range simpCallMap {
		simpToCalls := callIds2Names(simpCallMap[fromCall])
		trimToCalls := callIds2Names(sameModuleMap[fromCall])
		allCallMap[target.Syscalls[fromCall].Name] = CallMapItem{
			Module:      target.Syscalls[fromCall].Module,
			SimpVersion: simpToCalls,
			TrimVersion: trimToCalls,
		}
	}

	data, err := json.MarshalIndent(allCallMap, "", "\t")
	if err != nil {
		log.Fatalf("marshal fail %v", err)
	}
	err = osutil.WriteFile(outFileName, data)
	if err != nil {
		log.Fatalf("write data fail %v", err)
	}
}

func getSimpleResources(c *Syscall) (inpRescs []*ResourceDesc, outRescs []*ResourceDesc, hasFileInput bool) {
	inpDedup := make(map[*ResourceDesc]bool)
	outDedup := make(map[*ResourceDesc]bool)
	ForeachCallType(c, func(typ Type, ctx *TypeCtx) {
		if typ.Optional() {
			ctx.Stop = true
			return
		}

		switch typ1 := typ.(type) {
		case *ResourceType:
			if ctx.Dir != DirOut && !inpDedup[typ1.Desc] {
				inpDedup[typ1.Desc] = true
				inpRescs = append(inpRescs, typ1.Desc)
			}
			if ctx.Dir != DirIn && !outDedup[typ1.Desc] {
				outDedup[typ1.Desc] = true
				outRescs = append(outRescs, typ1.Desc)
			}
		case *BufferType:
			if ctx.Dir != DirOut && typ1.Kind == BufferFilename {
				hasFileInput = true
			}

		case *StructType, *UnionType:
			ctx.Stop = true
		}
	})
	return
}

func (target *Target) GenTarget2Relate() map[int][]int {
	callRelationMap := make(map[int][]int, len(target.Syscalls))
	for metaId, meta := range target.Syscalls {
		rcallMap := make(map[int]bool)
		for _, res := range meta.inputResources {
			for _, ctor := range res.Ctors {
				if ctor.Precise {
					rcallMap[ctor.Call] = true
				}
			}
		}
		rcalls := make([]int, 0, len(rcallMap))
		for k := range rcallMap {
			rcalls = append(rcalls, k)
		}
		callRelationMap[metaId] = rcalls
	}
	return callRelationMap
}

func (target *Target) GenRelate2Context() map[int][]int {
	relContextMap := make(map[int][]int, len(target.Syscalls))
	for metaId, meta := range target.Syscalls {
		for _, res := range meta.inputResources {
			for _, ctor := range res.Ctors {
				if ctor.Precise {
					relContextMap[ctor.Call] = append(relContextMap[ctor.Call], metaId)
				}
			}
		}
	}
	return relContextMap
}
```
