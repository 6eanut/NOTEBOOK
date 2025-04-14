#include <stdio.h>
#include <omp.h>
#define SIZE (2048 * 2048)  // 加括号避免宏展开问题

int a_array[SIZE], b_array[SIZE], a_plus_b_array[SIZE];

int main() {
    // 打印线程信息
    #pragma omp parallel
    {
        printf("Thread %d: hello_openmp\n", omp_get_thread_num());
    }

    // 初始化数组
    for (int i = 0; i < SIZE; ++i) {
        a_array[i] = i;
        b_array[i] = i + 1;
    }

    // 并行计算（自动划分迭代）
    #pragma omp parallel for
    for (int i = 0; i < SIZE; ++i) {
        a_plus_b_array[i] = a_array[i] + b_array[i];
    }

    printf("check: %d (expected: %d)\n", 
           a_plus_b_array[SIZE-1], 
           2 * SIZE - 1);  // 应输出 8388607
    return 0;
}