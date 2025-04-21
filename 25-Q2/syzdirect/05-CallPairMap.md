# CallPairMap

系统调用对：目标系统调用->相关系统调用们

```go
type CallPairMap map[int][]int
```

## 1 初始化

在启动Manager时会对系统调用对信息进行初始化。

### 1-1 初始化过程

CallPairFromFile函数会将指定文件中的信息转化为CallPairMap格式的系统调用对：

* 将json格式的系统调用对转换为string格式的系统调用对(RawCallPair struct)；
* 将string格式的系统调用对转换为用ID表示的系统调用对(tmpCallMap map)；
* 将tmpCallMap格式的系统调用对转换为CallPairMap格式的系统调用对(map[int]map[int]bool->map[int][]int)。

```go
// RawCallPair用string来描述系统调用对
type RawCallPair struct {
	Target string
	Relate []string
}

// 初始化CallPairMap，从json文件读取Tcall和Rcalls，并转化为CallPairMap格式
func CallPairFromFile(filename string, target *Target) CallPairMap {
	// 先将json转化成[]RawCallPair，Tcall和Rcall都用string来表示
	b, err := ioutil.ReadFile(filename)
	var rawCallPairs []RawCallPair
	err = json.Unmarshal(b, &rawCallPairs)

	// 将[]RawCallPair转化成tmpCallMap，Tcall和Rcall由用string表示转化成用ID表示，并用map去重
	tmpCallMap := make(map[int]map[int]bool, len(rawCallPairs))
	for _, rawCallPair := range rawCallPairs {
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

	// 将tmpCallMap转化成CallPairMap，将map转换成[]，并排序
	callPairMap := make(CallPairMap, len(tmpCallMap))
	for tcall, rcallMap := range tmpCallMap {
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

### 1-2 何时被初始化

CallPairMap是Manager结构体的一个成员变量，通过追踪CallPairFromFile函数，发现在启动Manager时，在RunManager函数中调用了CallPairFromFile函数。

## 2 读

在对语料库和选择表初始化时，会读系统调用对信息。

### 2-1 语料库

因为语料库中的程序必须包含目标系统调用，所以在初始化语料库时，会通过rawTcalls检查程序是否包含目标系统调用，起到筛选的作用；

通过GetRawTargetCalls函数读CallPairMap，获取rawTcalls，即只保留目标系统调用。

```go
// 获得Tcalls
// 将CallPairMap类型的cpMap转化成map[int]bool类型的rawTcalls，即只保留Tcall
func (cpMap CallPairMap) GetRawTargetCalls(target *Target) map[int]bool {
	rawTcalls := make(map[int]bool, len(cpMap))
	for tcall := range cpMap {
		// 处理带有 "_rf1" 或 "$tmp_rf1" 后缀的变异调用
		callName := target.Syscalls[tcall].Name
		if strings.HasSuffix(callName, "_rf1") {
			// 将变异调用还原为原始调用
			oriName := callName[:len(callName)-4]	
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
```

### 2-2 选择表

选择表的构建分两步：1.初始化系统调用(和syzkaller相同)，2.初始化系统调用对信息(syzdirect特有)。

系统调用对信息的初始化由EnableGo函数完成：

```go
// 选择表中系统调用对信息的存储形式是CallPairInfo数组
type CallPairInfo struct {
	Tcall int
	Rcall int
	Dists []uint32
	Prio int
}

// 依照CallPairMap(静态)和rpcCallPairMap(动态，包含dist)来对ChoiceTable的CallPairSelector做初始化
func (ct *ChoiceTable) EnableGo(cpMap CallPairMap, rpcCPMap RpcCallPairMap, corpus []*Prog, startTime time.Time, hitIndex uint32) {
	// 初始化 CallPairInfo 列表和infoIdxMap索引映射
	cpInfos := make([]CallPairInfo, 0, len(cpMap)*3)
	infoIdxMap := make(map[int]map[int]int, len(cpMap))
	allPrio := 0
	// 遍历CallPairMap的所有调用对（Tcall → []Rcall）
	for tcall, rcalls := range cpMap {
		if !ct.Enabled(tcall) {
			continue
		}
		rpcRcallMap, ok1 := rpcCPMap[tcall]
		tmp := append(rcalls, -1)
		rIdxMap := make(map[int]int)
		// 处理每个Rcall（包括无关联调用 Rcall=-1）
		for _, rcall := range tmp {
			if rcall != -1 && !ct.Enabled(rcall) {
				continue
			}
			hasAdd := false
			if ok1 {
				// 从 rpcCallPairMap 加载历史距离数据（如有）
				if dists2, ok2 := rpcRcallMap[rcall]; ok2 {
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
			// 默认优先级（无历史数据时），有关联prio=1，无关联prio=0
			if !hasAdd {			
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
	// 更新ChoiceTable中的CallPairSelector
	ct.GoEnable = true							
	ct.startTime = startTime
	ct.CallPairSelector.hitIndex = hitIndex
	ct.CallPairSelector.callPairInfos = cpInfos
	ct.CallPairSelector.prioSum = allPrio
	ct.CallPairSelector.infoIdxMap = infoIdxMap
}
```
