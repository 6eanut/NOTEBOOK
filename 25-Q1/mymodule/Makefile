KDIR := /home/syzkaller/linux
obj-m += mymodule.o

all:
        make -C $(KDIR) M=$(shell pwd) modules
clean:
        make -C $(KDIR) M=$(shell pwd) clean