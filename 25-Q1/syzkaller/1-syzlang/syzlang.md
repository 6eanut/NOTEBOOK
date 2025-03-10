# 系统调用描述语言(syzlang)

系统调用描述的伪正式语法：

```
syscallname "(" [arg ["," arg]*] ")" [type] ["(" attribute* ")"]
arg = argname type
argname = identifier
type = typename [ "[" type-options "]" ]
typename = "const" | "intN" | "intptr" | "flags" | "array" | "ptr" |
	   "string" | "filename" | "glob" | "len" |
	   "bytesize" | "bytesizeN" | "bitsize" | "vma" | "proc" |
	   "compressed_image"
type-options = [type-opt ["," type-opt]]
```

常见的 `type-opt`包括：

```
"opt" - the argument is optional (like mmap fd argument, or accept peer argument)
```

其余的 `type-opt`是针对特定类型的：

```
"const": integer constant, type-options:
	value, underlying type (one of "intN", "intptr")
"intN"/"intptr": an integer without a particular meaning, type-options:
	either an optional range of values (e.g. "5:10", or "100:200")
	or a reference to flags description (see below),
	or a single value
	optionally followed by an alignment parameter if using a range
"flags": a set of values, type-options:
	reference to flags description (see below), underlying int type (e.g. "int32")
"array": a variable/fixed-length array, type-options:
	type of elements, optional size (fixed "5", or ranged "5:10", boundaries inclusive)
"ptr"/"ptr64": a pointer to an object, type-options:
	direction (in/out/inout); type of the object
	ptr64 has size of 8 bytes regardless of target pointer size
"string": a zero-terminated memory buffer (no pointer indirection implied), type-options:
	either a string value in quotes for constant strings (e.g. "foo" or `deadbeef` for hex literal),
	or a reference to string flags (special value `filename` produces file names),
	optionally followed by a buffer size (string values will be padded with \x00 to that size)
"stringnoz": a non-zero-terminated memory buffer (no pointer indirection implied), type-options:
	either a string value in quotes for constant strings (e.g. "foo" or `deadbeef` for hex literal),
	or a reference to string flags,
"glob": glob pattern to match on the target files, type-options:
	a pattern string in quotes (syntax: https://golang.org/pkg/path/filepath/#Match)
	(e.g. "/sys/" or "/sys/**/*"),
	or include exclude glob too (e.g. "/sys/**/*:-/sys/power/state")
"fmt": a string representation of an integer (not zero-terminated), type-options:
	format (one of "dec", "hex", "oct") and the value (a resource, int, flags or proc)
	the resulting data is always fixed-size (formatted as "%020llu", "0x%016llx" or "%023llo", respectively)
"len": length of another field (for array it is number of elements), type-options:
	argname of the object
"bytesize": similar to "len", but always denotes the size in bytes, type-options:
	argname of the object
"bitsize": similar to "len", but always denotes the size in bits, type-options:
	argname of the object
"offsetof": offset of the field from the beginning of the parent struct, type-options:
	field
"vma"/"vma64": a pointer to a set of pages (used as input for mmap/munmap/mremap/madvise), type-options:
	optional number of pages (e.g. vma[7]), or a range of pages (e.g. vma[2-4])
	vma64 has size of 8 bytes regardless of target pointer size
"proc": per process int (see description below), type-options:
	value range start, how many values per process, underlying type
"compressed_image": zlib-compressed disk image
	syscalls accepting compressed images must be marked with `no_generate`
	and `no_minimize` call attributes. if the content of the decompressed image
	can be checked by a `fsck`-like command, use the `fsck` syscall attribute
"text": machine code of the specified type, type-options:
	text type (x86_real, x86_16, x86_32, x86_64, arm64)
"void": type with static size 0
	mostly useful inside of templates and varlen unions, can't be syscall argument
```

`attribute`包括：

```
"disabled": the call will not be used in fuzzing; useful to temporary disable some calls
	or prohibit particular argument combinations.
"timeout[N]": additional execution timeout (in ms) for the call on top of some default value
"prog_timeout[N]": additional execution timeout (in ms) for the whole program if it contains this call;
	if a program contains several such calls, the max value is used.
"ignore_return": ignore return value of this syscall in fallback feedback; need to be used for calls
	that don't return fixed error codes but rather something else (e.g. the current time).
"breaks_returns": ignore return values of all subsequent calls in the program in fallback feedback (can't be trusted).
"no_generate": do not try to generate this syscall, i.e. use only seed descriptions to produce it.
"no_minimize": do not modify instances of this syscall when trying to minimize a crashing program.
"fsck": the content of the compressed buffer argument for this syscall is a file system and the
    string argument is a fsck-like command that will be called to verify the filesystem
"remote_cover": wait longer to collect remote coverage for this call.
```

## Ints

* `int8`, `int16`, `int32`, `int64`：分别表示 1 字节、2 字节、4 字节和 8 字节的整数
* `intptr`：表示指针大小的整数
* 通过在类型后面添加 `be` 后缀，可以指定整数为大端模式（Big-Endian）
* 可以通过 `[min:max]` 或 `[min:max, align]` 的形式指定整数的取值范围
* 可以引用 `flags`或直接指定常量值作为第一个 `type-opt`
* 通过 `intN:N` 的形式可以定义一个 N 位的位字段
* 整数类型可以用作其他类型的基础类型，例如 `const`, `flags`, `len`, 和 `proc`

```
example_struct {
	f0	int8			# random 1-byte integer
	f1	const[0x42, int16be]	# const 2-byte integer with value 0x4200 (big-endian 0x42)
	f2	int32[0:100]		# random 4-byte integer with values from 0 to 100 inclusive
	f3	int32[1:10, 2]		# random 4-byte integer with values {1, 3, 5, 7, 9}
	f4	int64:20		# random 20-bit bitfield
	f5	int8[10]		# const 1-byte integer with value 10
	f6	int32[flagname]		# random 4-byte integer from the set of values referenced by flagname
}
```

## Structs

描述 `Structs`的语法：

```
structname "{" "\n"
	(fieldname type ("(" fieldattribute* ")")? (if[expression])? "\n")+
"}" ("[" attribute* "]")?
```

* 字段可以有独立于类型的属性，其中常见的方向属性包括 `in`, `out`, 和 `inout`，用于指定字段的数据流向

```
foo {
	field0	const[1, int32]	(in)
	field1	int32		(inout)
	field2	fd		(out)
}
```

* 可以通过 `if[expression]` 来指定字段是否包含在结构体中

```
foo {
	field0	int32
	field1	int32 (if[value[field0] == 0x1])
}
```

* 通过 `out_overlay` 属性，可以实现结构体的输入和输出布局分离
  * 在 `out_overlay` 字段之前的字段是输入字段
  * 从 `out_overlay` 字段开始的字段是输出字段
  * 输入和输出字段在内存中重叠（即都从结构体的起始位置开始）

```
foo {
    in0 const[1, int32]              # 输入字段
    in1 flags[bar, int8]             # 输入字段
    in2 ptr[in, string]              # 输入字段
    out0 fd (out_overlay)            # 输出字段，标记输入/输出布局分离
    out1 int32                       # 输出字段
}
```

`Structs`可以有以下属性，用方括号 `[]` 指定

* `packed`：表示结构体没有字段间的填充（padding），并且对齐方式为 1；类似于 GNU C 的 `__attribute__((packed))`；对齐方式可以通过 `align` 属性覆盖。
* `align[N]`：表示结构体的对齐方式为 `N`，并且会填充到 `N` 的倍数；填充的内容未指定（通常是零）；类似于 GNU C 的 `__attribute__((aligned(N)))`。
* `size[N]`：表示结构体会被填充到指定大小 `N`；填充的内容未指定（通常是零）。

## Unions

描述 `Unions`的语法：

```
unionname "[" "\n"
	(fieldname type (if[expression])? "\n")+
"]" ("[" attribute* "]")?
```

* 在模糊测试（fuzzing）过程中，`syzkaller` 会随机选择联合体中的一个选项作为当前的有效字段
* 可以通过 `if[expression]` 来指定某个字段是否可以被选中，这取决于其他字段的值

`Unions`可以有以下属性，用方括号 `[]`指定

* `varlen`：表示联合体的大小是当前所选选项的大小（动态决定）；如果没有这个属性，默认情况下联合体的大小是所有选项中最大的那个（类似于 C 语言中的联合体）
* `size[N]`：表示联合体会被填充到指定大小 `N`；填充的内容未指定（通常是零）

## Resources

> 资源代表需要从一个系统调用的输出传递到另一个系统调用的输入的值。
>
> 例如，close 系统调用需要先前由 open 或 pipe 系统调用返回的输入值（fd）。 为此，fd 被声明为一种资源。
>
> 这是一种模拟系统调用之间依赖关系的方法，因为将一个系统调用定义为资源的生产者，而将另一个系统调用定义为资源的消费者，就在它们之间定义了一种松散的排序方式。

描述 `Resources`的语法：

```
"resource" identifier "[" underlying_type "]" [ ":" const ("," const)* ]
```

* **`underlying_type`** ：资源的基础类型，可以是以下之一：
  * 基本整数类型：`int8`, `int16`, `int32`, `int64`, `intptr`
  * 另一个资源类型（用于建模继承关系，例如 `sock` 是 `fd` 的子类型）
* **`const`** ：可选的特殊值集合，用于表示资源的特殊状态（如无效值或默认值）。如果没有指定特殊值，默认使用 `0`

```
# fd 是一个基于 int32 的资源，具有三个特殊值：0xffffffffffffffff（表示无效文件描述符），AT_FDCWD（当前工作目录），以及 1000000（其他特殊值）
resource fd[int32]: 0xffffffffffffffff, AT_FDCWD, 1000000
# sock 是 fd 的子类型，表示套接字也是一种文件描述符
resource sock[fd]
# sock_unix 是 sock 的子类型，表示 Unix 套接字是一种特殊的套接字
resource sock_unix[sock]

# socket 系统调用返回一个 sock 类型的资源
socket(...) sock
# accept 系统调用接受一个 sock 类型的文件描述符作为输入，并返回一个新的 sock 类型资源
accept(fd sock, ...) sock
# listen 系统调用接受一个 sock 类型的文件描述符作为输入
listen(fd sock, backlog int32)
```

* 资源可以像其他数据类型一样使用，不一定必须由系统调用返回

```
# 定义了一个名为 my_resource 的资源，基础类型为 int32
resource my_resource[int32]

# request_producer 系统调用通过指针输出一个 my_resource 类型的值
request_producer(..., arg ptr[out, my_resource])
# request_consumer 系统调用通过指针输入/输出一个包含 my_resource 字段的结构体
request_consumer(..., arg ptr[inout, test_struct])

test_struct {
	...
	attr	my_resource
}
```

* 在更复杂的场景中，可以通过字段属性来进一步控制资源的生产与消费

```
# 定义了两个资源类型：my_resource_1 和 my_resource_2
resource my_resource_1[int32]
resource my_resource_2[int32]

# request_produce1_consume2 系统调用通过指针输入/输出一个包含 test_struct 的结构体
request_produce1_consume2(..., arg ptr[inout, test_struct])

test_struct {
	...
	# field0 被标记为 (out)，表示它是一个输出字段，生产 my_resource_1 类型的值
	field0	my_resource_1	(out)
	# field1 被标记为 (in)，表示它是一个输入字段，消费 my_resource_2 类型的值
	field1	my_resource_2	(in)
}
```

* 每个资源类型必须满足以下条件：(除了unions和optional pointers)
  * 至少由一个系统调用生产（作为输出）
  * 至少由一个系统调用消费（作为输入）

## Type Aliases

对于经常重复使用的复杂类型，可以使用以下语法给出简短的类型别名：

```
type identifier underlying_type
```

举个例子：

```
type signalno int32[0:65]
type net_port proc[20000, 4, int16be]
```

> 在任何情况下都可以使用类型别名来代替基础类型。 而且，类型别名也可以用作系统调用参数。 基础类型目前仅限于整数类型、ptr、ptr64、const、flags和 proc 类型。

有一些内置的 `Type Aliases`：

```
type bool8	int8[0:1]
type bool16	int16[0:1]
type bool32	int32[0:1]
type bool64	int64[0:1]
type boolptr	intptr[0:1]

type fileoff[BASE] BASE

type filename string[filename]

type buffer[DIR] ptr[DIR, array[int8]]
```

## Type Templates

声明语法如下：

```
type buffer[DIR] ptr[DIR, array[int8]]
type fileoff[BASE] BASE
type nlattr[TYPE, PAYLOAD] {
	nla_len		len[parent, int16]
	nla_type	const[TYPE, int16]
	payload		PAYLOAD
} [align_4]
```

使用语法如下：

```
syscall(a buffer[in], b fileoff[int64], c ptr[in, nlattr[FOO, int32]])
```

有一些内置的 `Type Templates`：

```
type optional[T] [
	val	T
	void	void
] [varlen]
```

说明：

```
type buffer[DIR] ptr[DIR, array[int8]]
```

* 定义了一个名为 `buffer` 的类型模板
* 参数 `DIR` 表示方向（如 `in`, `out`, 或 `inout`）
* 类型定义为一个指向字节数组的指针，方向由 `DIR` 决定
* **用途** ：用于表示输入/输出缓冲区

```
type fileoff[BASE] BASE
```

* 定义了一个名为 `fileoff` 的类型模板
* 参数 `BASE` 表示基础类型（如 `int32`, `int64` 等）
* 类型定义直接继承自 `BASE`
* **用途** ：用于表示文件偏移量，支持不同大小的基础类型

```
type nlattr[TYPE, PAYLOAD] {
    nla_len     len[parent, int16]
    nla_type    const[TYPE, int16]
    payload     PAYLOAD
} [align_4]
```

* 定义了一个名为 `nlattr` 的类型模板
* 参数 `TYPE` 和 `PAYLOAD` 分别表示属性类型和负载类型
* 类型定义为一个结构体，包含以下字段：
  * `nla_len`：表示长度字段，类型为 `len[parent, int16]`
  * `nla_type`：表示类型字段，值固定为 `TYPE`，类型为 `int16`
  * `payload`：表示负载字段，类型由 `PAYLOAD` 决定
* 属性 `[align_4]`：表示该结构体对齐方式为 4 字节
* **用途** ：用于描述网络层属性（Netlink Attributes）

```
syscall(a buffer[in], b fileoff[int64], c ptr[in, nlattr[FOO, int32]])
```

* `buffer[in]`：实例化 `buffer` 模板，方向为 `in`
* `fileoff[int64]`：实例化 `fileoff` 模板，基础类型为 `int64`
* `ptr[in, nlattr[FOO, int32]]`：定义一个指向 `nlattr` 结构体的指针，方向为 `in`，`nlattr` 的类型为 `FOO`，负载类型为 `int32`

```
type optional[T] [
	val	T
	void	void
] [varlen]
```

* 参数 `T` 表示字段的具体类型
* 定义为联合体（union），包含两个选项：
  * `val`：表示实际值，类型为 `T`
  * `void`：表示空值，类型为 `void`
* 属性 `[varlen]`：表示联合体的大小取决于当前所选选项的大小（动态决定）
* **用途** ：用于建模可选字段，例如某些系统调用中可能存在的可选参数

## Length

可以用 `len`、`bytesize`和 `bitsize`来指定结构体中的域或者命名参数的长度，例如：

```
# count 字段的值是 buf 指针所指向的数组长度
write(fd fd, buf ptr[in, array[int8]], count len[buf])

sock_fprog {
	len	len[filter, int16]
	filter	ptr[in, array[sock_filter]]
}
```

* `bytesizeN`类型用于表示以 N 字节为单位的长度，N可能是1，2，4，8
* 如果 `len`的参数是指针，那么使用被指向参数的长度
* 要表示父结构体的长度，可以使用 `len[parent,int8]`。当结构相互嵌入时，要表示上级父类的长度，可以指定特定父类的类型名称

```
s1 {
    f0      len[s2]  # length of s2
}

s2 {
    f0      s1
    f1      array[int32]
    f2      len[parent, int32]	# 父结构体s1的长度
}
```

`len` 的参数可以是一个路径表达式，允许更复杂的地址引用：

* 路径表达式类似于 C 语言中的字段引用
* 支持引用父元素和兄弟元素
* 特殊引用 `syscall` 可以直接引用系统调用的参数

```
s1 {
	a	ptr[in, s2]
	b	ptr[in, s3]
	c	array[int8]
}

s2 {
	d	array[int8]
}

s3 {
# This refers to the array c in the parent s1.
	e	len[s1:c, int32]
# This refers to the array d in the sibling s2.
	f	len[s1:a:d, int32]
# This refers to the array k in the child s4.
	g	len[i:j, int32]
# This refers to syscall argument l.
	h	len[syscall:l, int32]
	i	ptr[in, s4]
}

s4 {
	j	array[int8]
}

foo(k ptr[in, s1], l ptr[in, array[int8]])
```

## Proc

`proc` 类型的主要目的是为每个执行器（executor）生成一组独立的整数值范围，以避免不同进程之间的值冲突，用于表示每个进程独有的整数值

最简单的例子是端口号。`proc[20000,4,int16be]`类型意味着我们想要从 `20000`开始生成一个 `int16be`整数，并为每个进程分配 `4`个值。因此，执行器编号 `n`将获得 `[20000+n*4,20000+(n+1)*4)`范围内的值。

## Integer Constants

整型常量可以指定为十进制字面值、以 `0x`为前缀的十六进制字面值、以 `'`包围的字符字面值，或者从内核头文件中提取或由define指令定义的符号常量。例如:

```
# 表示一个值为 10 的整数常量，const[-10] 表示一个值为 -10 的整数常量
foo(a const[10], b const[-10])
# 表示一个值为 0xabcd（即十进制的 43981）的整数常量
foo(a const[0xabcd])
# 表示一个范围为从字符 'a' 到 'z' 的整数常量，具体值是这些字符的 ASCII 编码
foo(a int8['a':'z'])
# 表示使用符号常量 PATH_MAX 的值，该值通常是从内核头文件中提取的符号常量，或者通过 define 指令定义的常量
foo(a const[PATH_MAX])
foo(a int32[PATH_MAX])
# define 指令允许用户定义新的符号常量，支持简单的数学运算
foo(a ptr[in, array[int8, MY_PATH_MAX]])
define MY_PATH_MAX	PATH_MAX + 2
```

## Conditional fields

### In structures

条件字段允许根据某些条件动态地决定结构体中的某个字段是否应该被包含或省略：

```
header_fields {
  magic       const[0xabcd, int16]
  haveInteger int8
} [packed]

packet {
  header  header_fields
  # 如果 header.haveInteger == 1，则 integer 字段会被包含；否则，integer 字段会被忽略
  integer int64  (if[value[header:haveInteger] == 0x1])
  body    array[int8]
} [packed]

some_call(a ptr[in, packet])
```

当条件字段存在时，结构体的内存布局会根据条件动态调整，例如：

* 当 `header.haveInteger == 1`：

```
| header_files.magic = 0xabcd | header_files.haveInteger = 0x1 | integer | body |
```

* 当 `header.haveInteger != 1`：

```
| header_files.magic = 0xabcd | header_files.haveInteger = 0x0 | body |
```

> 每个条件字段都假定为可变长度，该字段所属的结构体也是如此。当可变长度字段出现在结构体中间时，必须用`[packed]`标记该结构体。

禁止使用位域的条件，但可以在条件中引用位域：

```
struct {
  f0 int
  f1 int:3 (if[value[f0] == 0x1])  # It will not compile.
}

struct {
  f0 int:1
  f1 int:7
  f2 int   (if[value[f0] == value[f1]])
} [packed]
```

### In unions

```
struct {
  type int
  body alternatives
}

alternatives [
  int     int64 (if[value[struct:type] == 0x1])
  arr     array[int64, 5] (if[value[struct:type] == 0x2])
  default int32
] [varlen]

some_call(a ptr[in, struct])
```

在这个例子中，将根据 `type`字段的值选择 `union`选项。例如，如果 `type`是 `0x1`，那么它可以是 `int`或 `default`；如果 `type`是 `0x2`，那么它可以是 `arr`或 `default`；如果 `type`既不是 `0x1`也不是 `0x2`，那么它只可能是 `default`。

为了确保始终可以构造`union`，最后一个`union`字段必须始终没有条件。因此，以下定义将无法编译：

```
alternatives [
  int int64 (if[value[struct:type] == 0x1])
  arr array[int64, 5] (if[value[struct:type] == 0x1])
] [varlen]
```

### Expression Syntax

目前，只支持==，!=，&，||操作符。以下是一些例子：

```
sub_struct {
  f0 int
  # Reference a field in a parent struct.
  f1 int (if[value[struct:f2]]) # Same as if[value[struct:f2] != 0].
}

struct {
  f2 int
  f3 sub_struct
  f4 int (if[value[f2] == 0x2]) # Reference a sibling field.
  f5 int (if[value[f3:f0] == 0x1]) # Reference a nested field.
  f6 int (if[value[f3:f0] == 0x1 || value[f3:f0] == 0x2]) # Reference a nested field which either equals to 0x1 or 0x2.
} [packed]

call(a ptr[in, struct])
```

引用的字段必须是整数类型，并且在它的路径中不能有条件字段。例如，以下描述将无法编译：

```
struct {
  f0 int
  f1 int (if[value[f0] == 0x1])
  f2 int (if[value[f1] == 0x1])
}
```

也可以在表达式中引用常量：

```
struct {
  f0 int
  f1 int
  f2 int (if[value[f0] & SOME_CONST == OTHER_CONST])
}
```

## Meta

`meta` 指令用于在描述文件中指定元信息（meta-information），这些元信息会影响整个文件的行为。通过 `meta` 指令，可以控制文件的提取行为、适用的架构范围等。以下是一些用法：

```
# 告诉make extract不extract此文件的常量
meta noextract
# 限制该文件仅适用于指定的架构
meta arches["arch1", "arch2"]
```

## Misc

* `include` 指令用于引用 Linux 内核头文件
* `incdir` 指令用于指定自定义的 Linux 内核头文件目录
* `define` 指令用于定义符号常量值
* 有pseudo system calls
