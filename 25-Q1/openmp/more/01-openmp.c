#include <stdio.h>
#include <omp.h>
#define SIZE 2048 * 2048
int a_array[SIZE], b_array[SIZE], a_plus_b_array[SIZE];
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