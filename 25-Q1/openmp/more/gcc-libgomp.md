# 从汇编代码理解OpenMP

> 当用C语言去写一个程序的时候，一般来讲是会放在单个CPU上去运行，但若使用OpenMP，则可以将程序并行执行，那么具体是怎么实现将程序并行执行的？
>
> 一般来讲，并行化的部分是循环的部分，即将一个循环的任务拆分成多个互不相关的子任务，然后分配到多个CPU上去执行，所以这里会用一个简单的例子来解释OpenMP是如何将程序并行化的。

## 内存的分布

硬件内存到软件内存往往会采用页管理机制，即将实际的内存划分成多个页，而后当程序运行时，操作系统会为进程分配若干页，在每个进程的内存中，又会分为程序段、数据段、堆和栈，其中数据段又分为.data和.bss段。

```c
#include <stdio.h>
char bss_var[20];
char data_var[4] = "good";
int main()
{
    char stack_var[6] = "hello";
    char *heap_var = (char *)malloc(sizeof(char) * 5);
    return 0;
}
```

```asm
	.file	"memory.c"
	.option nopic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.globl	bss_var
	.bss
	.align	3
	.type	bss_var, @object
	.size	bss_var, 20
bss_var:
	.zero	20
	.globl	data_var
	.section	.sdata,"aw"
	.align	3
	.type	data_var, @object
	.size	data_var, 4
data_var:
	.ascii	"good"
	.text
	.align	1
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-32
	sd	ra,24(sp)
	sd	s0,16(sp)
	addi	s0,sp,32
	li	a5,1819041792
	addi	a5,a5,1384
	sw	a5,-32(s0)
	li	a5,111
	sh	a5,-28(s0)
	li	a0,5
	call	malloc
	mv	a5,a0
	sd	a5,-24(s0)
	li	a5,0
	mv	a0,a5
	ld	ra,24(sp)
	ld	s0,16(sp)
	addi	sp,sp,32
	jr	ra
	.size	main, .-main
	.ident	"GCC: (GNU) 12.3.1 (openEuler 12.3.1-30.oe2403)"
	.section	.note.GNU-stack,"",@progbits
```

数据段存放的是全局变量或静态变量，bss是未初始化的，data是初始化了的。栈存放局部变量，堆存放malloc申请的内存。

```
高地址
┌───────────────────────┐
│        栈（Stack）     │ ← 由高地址向低地址增长
├───────────────────────┤
│           ↓           │
│           ↑           │
├───────────────────────┤
│         堆（Heap）     │ ← 由低地址向高地址增长
├───────────────────────┤
│    BSS 段（未初始化数据）│
├───────────────────────┤
│    DATA 段（初始化数据） │
├───────────────────────┤
│     TEXT 段（代码段）   │
└───────────────────────┘
低地址
```

## 一个例子

从C语言程序到汇编程序，在汇编程序中分析OpenMP，然后查看libgomp的源码，最后引到pthread

```c
#include <stdio.h>
#include <omp.h>
#define SIZE 2048 * 2048
int a_array[SIZE], b_array[SIZE], a_plus_b_array[SIZE];		// 如果以局部变量的方式存在，那么会因为栈溢出导致段错误，因为在Linux中，默认线程栈大小为8MB，可以通过ulimit -s查看
int main()
{
#pragma omp parallel
    {
        printf("hello_openmp\n");
    }

    // init
    for (int i = 0; i < SIZE; ++i)
    {
        a_array[i] = i;
        b_array[i] = i + 1;
    }

// compute
#pragma omp parallel
    {
        int sum = omp_get_num_threads();
        int id = omp_get_thread_num();
#pragma omp for
        for (int i = id; i < SIZE; i += sum)
        {
            a_plus_b_array[i] = a_array[i] + b_array[i];
        }
    }

    printf("check : %d\n", a_plus_b_array[SIZE - 1]);
    return 0;
}
```

汇编代码

```asm
	.file	"openmp.c"
	.option nopic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"
	.attribute unaligned_access, 0
	.attribute stack_align, 16
	.text
	.globl	a_array
	.bss
	.align	3
	.type	a_array, @object
	.size	a_array, 16777216
a_array:
	.zero	16777216
	.globl	b_array
	.align	3
	.type	b_array, @object
	.size	b_array, 16777216
b_array:
	.zero	16777216
	.globl	a_plus_b_array
	.align	3
	.type	a_plus_b_array, @object
	.size	a_plus_b_array, 16777216
a_plus_b_array:
	.zero	16777216
	.section	.rodata
	.align	3
.LC0:
	.string	"check : %d\n"
	.text
	.align	1
	.globl	main
	.type	main, @function
main:
	addi	sp,sp,-32
	sd	ra,24(sp)
	sd	s0,16(sp)
	addi	s0,sp,32
	li	a3,0
	li	a2,0
	li	a1,0
	lui	a5,%hi(main._omp_fn.0)
	addi	a0,a5,%lo(main._omp_fn.0)
	call	GOMP_parallel
	sw	zero,-20(s0)
	j	.L2
.L3:
	lui	a5,%hi(a_array)
	addi	a4,a5,%lo(a_array)
	lw	a5,-20(s0)
	slli	a5,a5,2
	add	a5,a4,a5
	lw	a4,-20(s0)
	sw	a4,0(a5)
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sext.w	a4,a5
	lui	a5,%hi(b_array)
	addi	a3,a5,%lo(b_array)
	lw	a5,-20(s0)
	slli	a5,a5,2
	add	a5,a3,a5
	sw	a4,0(a5)
	lw	a5,-20(s0)
	addiw	a5,a5,1
	sw	a5,-20(s0)
.L2:
	lw	a5,-20(s0)
	sext.w	a4,a5
	li	a5,4194304
	blt	a4,a5,.L3
	li	a3,0
	li	a2,0
	li	a1,0
	lui	a5,%hi(main._omp_fn.1)
	addi	a0,a5,%lo(main._omp_fn.1)
	call	GOMP_parallel
	lui	a5,%hi(a_plus_b_array)
	addi	a4,a5,%lo(a_plus_b_array)
	li	a5,16777216
	add	a5,a4,a5
	lw	a5,-4(a5)
	mv	a1,a5
	lui	a5,%hi(.LC0)
	addi	a0,a5,%lo(.LC0)
	call	printf
	li	a5,0
	mv	a0,a5
	ld	ra,24(sp)
	ld	s0,16(sp)
	addi	sp,sp,32
	jr	ra
	.size	main, .-main
	.section	.rodata
	.align	3
.LC1:
	.string	"hello_openmp"
	.text
	.align	1
	.type	main._omp_fn.0, @function
main._omp_fn.0:
	addi	sp,sp,-32
	sd	ra,24(sp)
	sd	s0,16(sp)
	addi	s0,sp,32
	sd	a0,-24(s0)
	lui	a5,%hi(.LC1)
	addi	a0,a5,%lo(.LC1)
	call	puts
	ld	ra,24(sp)
	ld	s0,16(sp)
	addi	sp,sp,32
	jr	ra
	.size	main._omp_fn.0, .-main._omp_fn.0
	.align	1
	.type	main._omp_fn.1, @function
main._omp_fn.1:
	addi	sp,sp,-80
	sd	ra,72(sp)
	sd	s0,64(sp)
	sd	s1,56(sp)
	sd	s2,48(sp)
	sd	s3,40(sp)
	addi	s0,sp,80
	sd	a0,-72(s0)
	call	omp_get_num_threads
	mv	a5,a0
	sw	a5,-56(s0)
	call	omp_get_thread_num
	mv	a5,a0
	sw	a5,-60(s0)
	lw	s2,-60(s0)
	lw	s1,-56(s0)
	call	omp_get_num_threads
	mv	a5,a0
	mv	s3,a5
	call	omp_get_thread_num
	mv	a5,a0
	mv	a3,a5
	li	a5,4194304
	addiw	a5,a5,-1
	addw	a5,s1,a5
	sext.w	a5,a5
	subw	a5,a5,s2
	sext.w	a5,a5
	divw	a5,a5,s1
	sext.w	a4,a5
	divw	a5,a4,s3
	sext.w	a5,a5
	remw	a4,a4,s3
	sext.w	a4,a4
	mv	a1,a3
	mv	a2,a4
	blt	a1,a2,.L7
.L10:
	mulw	a3,a5,a3
	sext.w	a3,a3
	addw	a4,a3,a4
	sext.w	a4,a4
	addw	a5,a4,a5
	sext.w	a5,a5
	mv	a2,a4
	mv	a3,a5
	bge	a2,a3,.L11
	mulw	a4,a4,s1
	sext.w	a4,a4
	addw	a4,a4,s2
	sw	a4,-52(s0)
	mulw	a5,a5,s1
	sext.w	a5,a5
	addw	a5,a5,s2
	sext.w	a2,a5
.L9:
	lui	a5,%hi(a_array)
	addi	a4,a5,%lo(a_array)
	lw	a5,-52(s0)
	slli	a5,a5,2
	add	a5,a4,a5
	lw	a4,0(a5)
	lui	a5,%hi(b_array)
	addi	a3,a5,%lo(b_array)
	lw	a5,-52(s0)
	slli	a5,a5,2
	add	a5,a3,a5
	lw	a5,0(a5)
	addw	a5,a4,a5
	sext.w	a4,a5
	lui	a5,%hi(a_plus_b_array)
	addi	a3,a5,%lo(a_plus_b_array)
	lw	a5,-52(s0)
	slli	a5,a5,2
	add	a5,a3,a5
	sw	a4,0(a5)
	lw	a5,-52(s0)
	addw	a5,a5,s1
	sw	a5,-52(s0)
	lw	a5,-52(s0)
	sext.w	a5,a5
	mv	a4,a2
	blt	a5,a4,.L9
	j	.L11
.L7:
	li	a4,0
	addiw	a5,a5,1
	sext.w	a5,a5
	j	.L10
.L11:
	nop
	ld	ra,72(sp)
	ld	s0,64(sp)
	ld	s1,56(sp)
	ld	s2,48(sp)
	ld	s3,40(sp)
	addi	sp,sp,80
	jr	ra
	.size	main._omp_fn.1, .-main._omp_fn.1
	.ident	"GCC: (GNU) 12.3.1 (openEuler 12.3.1-30.oe2403)"
	.section	.note.GNU-stack,"",@progbits
```

可以看到汇编代码中有两个openmp函数，分别是main.ompfn.0和main.ompfn.1，这两个函数对应的就是要并行化的代码块。

然后在main函数里面，会将这两个ompfn的地址作为参数传递给GOMP_parallel函数，该函数是libgomp运行时库的函数，第一个参数是并行函数的地址，第二个参数是传递给并行函数的参数，第三个参数是线程数，第四个参数是标志位用于控制并行行为。

通过https://gcc.gnu.org/git/gcc.git可以查看gcc源代码，找到libgomp目录，以及GOMP_parallel的定义

```c
void GOMP_parallel (void (*fn) (void *), void *data, unsigned num_threads, unsigned int flags)
{
  num_threads = gomp_resolve_num_threads (num_threads, 0);
  gomp_team_start (fn, data, num_threads, flags, gomp_new_team (num_threads), NULL);		// 创建线程池，依赖pthread
  fn (data);											// 执行并行代码块
  ialias_call (GOMP_parallel_end) ();								// 同步
}
```

gomp_team_start会调用pthread_create来创建线程
