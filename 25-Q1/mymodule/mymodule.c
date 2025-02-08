#include<linux/module.h>
#include<linux/init.h>

static int __init my_module_init(void){
    printk("hello mymodule\n");
    return 0;
}

static void __exit my_module_exit(void){
    printk("goodbye mymodule\n");
}

module_init(my_module_init);
module_exit(my_module_exit);
MODULE_LICENSE("GPL");