#include <stdio.h>
char bss_var[20];
char data_var[4] = "good";
int main()
{
    char stack_var[6] = "hello";
    char *heap_var = (char *)malloc(sizeof(char) * 5);
    return 0;
}