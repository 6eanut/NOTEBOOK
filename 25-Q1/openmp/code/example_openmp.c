#include <stdio.h>
#include <omp.h>
int main()
{
    printf("#pragma omp parallel\n");
#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("hello %d\n", ID);
        printf("goodbye %d\n", ID);
    }
    printf("\n");

    printf("#pragma omp parallel num_threads(2)\n");
#pragma omp parallel num_threads(2)
    {
        int ID = omp_get_thread_num();
        printf("hello %d\n", ID);
        printf("goodbye %d\n", ID);
    }
    printf("\n");

    printf("#pragma omp barrier\n");
#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("hello %d\n", ID);
#pragma omp barrier
        printf("goodbye %d\n", ID);
    }
    printf("\n");

    // This can happen because there will be competition for data
    // nothing : x[0] = 0 , x[1] = 1
    // 0 : x[0] = 1
    // 1 : x[1] = 2
    // 3 : x[1] = 2
    // 2 : x[0] = 2
    // x[0] = 2 , x[1] = 2
    int x[2] = {0, 1};
    printf("nothing : x[0] = 0 , x[1] = 1\n");
#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        x[ID % 2]++;
#pragma omp barrier
        printf("%d : x[%d] = %d\n", ID, ID % 2, x[ID % 2]);
    }
    printf("x[0] = %d , x[1] = %d\n", x[0], x[1]);
    printf("\n");

    // The granularity is the variable x
    x[0] = 0, x[1] = 1;
    printf("critical : x[0] = 0 , x[1] = 1\n");
#pragma omp parallel
    {
        int ID = omp_get_thread_num();
#pragma omp critical
        x[ID % 2]++;
#pragma omp barrier
        printf("%d : x[%d] = %d\n", ID, ID % 2, x[ID % 2]);
    }
    printf("x[0] = %d , x[1] = %d\n", x[0], x[1]);
    printf("\n");

    // The granularity is the variable x[i]
    x[0] = 0, x[1] = 1;
    printf("atomic : x[0] = 0 , x[1] = 1\n");
#pragma omp parallel
    {
        int ID = omp_get_thread_num();
#pragma omp atomic
        x[ID % 2]++;
#pragma omp barrier
        printf("%d : x[%d] = %d\n", ID, ID % 2, x[ID % 2]);
    }
    printf("x[0] = %d , x[1] = %d\n", x[0], x[1]);
    printf("\n");

    printf("#pragma omp parallel for\n");
#pragma omp parallel for
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int ID = omp_get_thread_num();
            printf("for:%d -- for:%d-- ID:%d\n", i, j, ID);
        }
    }
    printf("\n");

    printf("#pragma omp parallel for collapse(2)\n");
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int ID = omp_get_thread_num();
            printf("for:%d -- for:%d-- ID:%d\n", i, j, ID);
        }
    }
    printf("\n");

    // This can happen:
    // #pragma omp parallel for
    // sum = 0
    // sum = 2
    int sum = 0;
    printf("#pragma omp parallel for\n");
    printf("sum = %d\n", sum);
#pragma omp parallel for
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int ID = omp_get_thread_num();
            sum++;
        }
    }
    printf("sum = %d\n", sum);

    sum = 0;
    printf("#pragma omp parallel for reduction(+:sum)\n");
    printf("sum = %d\n", sum);
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int ID = omp_get_thread_num();
            sum++;
        }
    }
    printf("sum = %d\n", sum);
    return 0;
}