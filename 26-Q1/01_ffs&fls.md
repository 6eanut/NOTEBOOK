# ffs, fls, __ffs, __fls

好久没写notebook了，最近在做的事情是为google/syzkaller添加Linux RISC-V KVM Fuzz的[相关实现](https://github.com/google/syzkaller/commits/master/?author=6eanut)，幸运地找到了一些bug，并且部分已经被社区[接收了](https://github.com/search?q=repo%3Atorvalds%2Flinux+Jiakai+Xu&type=commits)。

|    | **补丁修复位置**                      | **补丁**                                                                                                                                                                                                                                                                                     | **状态**           |
| -- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| 15 | arch/riscv/kvm                              | **BUG: unable to handle kernel paging request in kvm_riscv_vcpu_pmu_ctr_cfg_match**                                                                                                                                                                                                          | 待分析                   |
| 14 | mm                                          | **[[BUG] WARNING in unlink_anon_vmas()](https://lore.kernel.org/linux-mm/CAFb8wJvRhatRD-9DVmr5v5pixTMPEr3UKjYBJjCd09OfH55CKg@mail.gmail.com/)**, **[[PATCH mm-hotfixes] mm/rmap: clear vma-&gt;anon_vma on error](https://lore.kernel.org/linux-mm/20260318122632.63404-1-ljs@kernel.org/)** | 已被maintainer确认并修复 |
| 13 | arch/riscv/kvm                              | **[[PATCH] RISC-V: KVM: Fix double-free of sdata in kvm_pmu_clear_snapshot_area()](https://lore.kernel.org/linux-riscv/20260318092956.708246-1-xujiakai2025@iscas.ac.cn/T/#u)**                                                                                                                 | 待审阅                   |
| 12 | arch/riscv/kvm, tools/testing/selftests/kvm | **[[PATCH v5 0/2] RISC-V: KVM: Fix array out-of-bounds in firmware counter reads](https://lore.kernel.org/kvm-riscv/20260316014533.2312254-1-xujiakai2025@iscas.ac.cn/T/#t)**                                                                                                                   | 已审阅                   |
| 11 | arch/riscv/kvm                              | **[[PATCH] RISC-V: KVM: Fix potential UAF in kvm_riscv_aia_imsic_has_attr()](https://lore.kernel.org/linux-riscv/20260304080804.2281721-1-xujiakai2025@iscas.ac.cn/)**                                                                                                                          | 进入主线                 |
| 10 | arch/riscv/kvm                              | **[[PATCH v3] RISC-V: KVM: Fix use-after-free in kvm_riscv_aia_aplic_has_attr()](https://lore.kernel.org/linux-riscv/20260302132703.1721415-1-xujiakai2025@iscas.ac.cn/T/#u)**                                                                                                                  | 进入主线                 |
| 9  | arch/riscv/kvm                              | **[[PATCH] RISC-V: KVM: Change imsic-&gt;vsfile_lock from rwlock_t to raw_spinlock_t](https://lore.kernel.org/linux-riscv/20260131025800.1550692-1-xujiakai2025@iscas.ac.cn/T/#u)**                                                                                                             | 讨论中                   |
| 8  | arch/riscv/kvm                              | **[[PATCH v2] RISC-V: KVM: Fix null pointer dereference in kvm_riscv_vcpu_aia_rmw_topei()](https://lore.kernel.org/linux-riscv/20260226085119.643295-1-xujiakai2025@iscas.ac.cn/T/#u)**                                                                                                         | 进入主线                 |
| 7  | arch/riscv/kvm                              | **[[PATCH v2] RISC-V: KVM: Fix use-after-free in kvm_riscv_gstage_get_leaf()](https://lore.kernel.org/linux-riscv/20260202040059.1801167-1-xujiakai2025@iscas.ac.cn/T/#u)**                                                                                                                     | 进入主线                 |
| 6  | arch/riscv/kvm                              | **[[PATCH] RISC-V: KVM: Skip IMSIC update if vCPU IMSIC state is not initialized](https://lore.kernel.org/linux-riscv/20260127084313.3496485-1-xujiakai2025@iscas.ac.cn/)**                                                                                                                     | 进入主线                 |
| 5  | arch/riscv/kvm                              | **[[PATCH] RISC-V: KVM: Fix null pointer dereference in kvm_riscv_aia_imsic_rw_attr()](https://lore.kernel.org/linux-riscv/20260127072219.3366607-1-xujiakai2025@iscas.ac.cn/)**                                                                                                                | 进入主线                 |
| 4  | arch/riscv/kvm                              | **[[PATCH v4] RISC-V: KVM: Fix null pointer dereference in kvm_riscv_aia_imsic_has_attr()](https://lore.kernel.org/linux-riscv/20260125143344.2515451-1-xujiakai2025@iscas.ac.cn/)**                                                                                                            | 进入主线                 |
| 3  | arch/riscv/kvm, tools/testing/selftests/kvm | **[[PATCH v10 0/3] RISC-V: KVM: Validate SBI STA shmem alignment](https://lore.kernel.org/linux-riscv/20260303010859.1763177-1-xujiakai2025@iscas.ac.cn/T/#t)**                                                                                                                                 | 已审阅                   |
| 2  | arch/riscv/kernel                           | **[[PATCH v2] riscv: fix KUnit test_kprobes crash when building with Clang](https://lore.kernel.org/linux-riscv/20251226032317.1523764-1-jiakaiPeanut@gmail.com/)**                                                                                                                             | 进入主线                 |
| 1  | arch/riscv/kernel                           | **[[PATCH] riscv: stacktrace: Disable KASAN checks for non-current tasks](https://lore.kernel.org/linux-riscv/20251022072608.743484-1-zhangchunyan@iscas.ac.cn/)**                                                                                                                              | 进入主线                 |

好，言归正传，今天在分析 `BUG: unable to handle kernel paging request in kvm_riscv_vcpu_pmu_ctr_cfg_match` 时遇到了 `ffs`, `fls`, `__ffs`, `__fls`这些helper函数，一开始搞不明白，后面在LLM的帮助下才知道是怎么回事。

## 1 先看源码

这些函数都是位操作工具函数，定义在`arch/riscv/include/asm/bitops.h`里面：

```c
// 没必要看源码，看注释就行
/**
 * ffs - find first set bit in a word
 * @x: the word to search
 *
 * This is defined the same way as the libc and compiler builtin ffs routines.
 *
 * ffs(value) returns 0 if value is 0 or the position of the first set bit if
 * value is nonzero. The first (least significant) bit is at position 1.
 */
#define ffs(x) (__builtin_constant_p(x) ? __builtin_ffs(x) : variable_ffs(x))

/**
 * fls - find last set bit in a word
 * @x: the word to search
 *
 * This is defined in a similar way as ffs, but returns the position of the most
 * significant set bit.
 *
 * fls(value) returns 0 if value is 0 or the position of the last set bit if
 * value is nonzero. The last (most significant) bit is at position 32.
 */
#define fls(x)							\
({								\
	typeof(x) x_ = (x);					\
	__builtin_constant_p(x_) ?				\
	 ((x_ != 0) ? (32 - __builtin_clz(x_)) : 0)		\
	 :							\
	 variable_fls(x_);					\
})

/**
 * __ffs - find first set bit in a long word
 * @word: The word to search
 *
 * Undefined if no set bit exists, so code should check against 0 first.
 */
#define __ffs(word)				\
	(__builtin_constant_p(word) ?		\
	 (unsigned long)__builtin_ctzl(word) :	\
	 variable__ffs(word))

/**
 * __fls - find last set bit in a long word
 * @word: the word to search
 *
 * Undefined if no set bit exists, so code should check against 0 first.
 */
#define __fls(word)							\
	(__builtin_constant_p(word) ?					\
	 (unsigned long)(BITS_PER_LONG - 1 - __builtin_clzl(word)) :	\
	 variable__fls(word))
```

好了，翻译一下：`ffs`指的是Find First Set bit, `fls`指的是Find Last Set bit; 加上`__`之后，代表着最低为是索引0，否则是1.

## 2 看一个例子

```demo
data = 0b00101000
ffs(data) = 4
fls(data) = 6
__ffs(data) = 3 
__fls(data) = 5
```
