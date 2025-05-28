# sysMaster

源码仓库：[https://gitee.com/openeuler/sysmaster](https://gitee.com/openeuler/sysmaster)

## 20250528

硬件CPU在通电后会进入Reset State，而后会执行Firmware，即下面所说的BIOS，执行完Firmware后会去磁盘上加载第0号扇区，即前512字节的内容，这段内容是Master Boot Record，这个MBR会去加载内核等后面的内容。

在X86里面，MBR的最后内容会设置成0x55 0xAA，以此来检验这块数据是否有效。Firmware会把MBR的内容加载到内存的0x7c00位置。

在RISC-V里面，CPU进入reset state之后，会执行OpenSBI等firmware，然后会去加载内核，固件的入口一般规定在0x80000000.

---

## 0-背景

Linux的启动流程大致如下:

1. BIOS：硬件检测，查找并加载磁盘上的MBR
2. MBR：存储BootLoader信息，加载GRUB
3. GRUB：查找并加载Kernal
4. Kernal：装载驱动，挂载rootfs，执行/sbin/init
5. Init：OS初始化，执行runlevel相关程序
6. Runlevel：启动指定级别的服务

> MBR：MBR（Master Boot Record）是硬盘上一段重要的引导代码区域，位于硬盘的第一个扇区（通常是第0号扇区，即第512字节的位置）。这个扇区包含了引导加载程序（boot loader）以及分区表（partition table），用于告诉计算机操作系统的位置。
>
> GRUB：GRUB（GNU GRand Unified Bootloader）是一种广泛使用的引导加载程序（boot loader），主要用于 Unix-like 操作系统，如 Linux 和 BSD。

Init是OS的初始化过程，最常见的就是systemd~~，但是其代码量大并且性能一般，所以有了sysMaster~~

## 1-sysMaster

sysMaster共有init、core、exts三个部分：

* init：1号进程，LOC 1k左右
* core：systemd原有核心功能，插件化、崩溃快速自愈
* exts：抽屉式、解耦各组件

### 1-0 init

```rust
// main函数位于sysmaster/init/main.rs
fn main() -> std::io::Result<()> {
    setup_mount_early();

    log::init_log_to_kmsg_console("sysmaster-init", log::Level::Info);

    prepare_init();

    reset_all_signal_handlers();
    install_crash_handler();

    let mut rt = Runtime::new()?;
    rt.set_state(InitState::Running);

    rt.runloop()?;

    if rt.is_reexec() {
        rt.reexec();
    }
    shutdown_init();
    Ok(())
}
```

* setup_mount_early

  * 在/sysmaster/init/mount.rs文件中[定义](image/sysMaster/setup_mount_early.png)
  * 功能：挂载文件系统/sys /proc /dev /run
  * /sys和/proc对于获取系统和进程信息非常有用；/dev和/run对于存放设备文件和运行时数据很关键
* init_log_to_kmsg_console

  * 在/sysmaster/libs/log/src/lib.rs文件中[定义](image/sysMaster/init_log_to_kmsg_console.png)
  * 功能：初始化日志记录器
* prepare_init

  * 在/sysmaster/init/main.rs文件中[定义](image/sysMaster/prepare_init.png)
  * 功能：确保daemon以正确的权限和状态运行
  * 设置文件权限掩码、确保进程以超级用户权限运行、确保pid是1
* reset_all_signal_handlers

  * 在/sysmaster/init/main.rs文件中[定义](image/sysMaster/reset_all_signal_handlers.png)
  * 功能：设置信号处理程序为忽略，除了SIGKILL和SIGSTOP
  * 确保daemon能够稳定地运行，并且不会因为外部信号而导致不可预测的行为
* install_crash_handler

  * 在/sysmaster/init/main.rs文件中[定义](image/sysMaster/install_crash_handler.png)
  * 功能：安装崩溃处理程序
  * 这些都是严重的错误信号，需要特殊处理
* Runtime

  * 在/sysmaster/init/runtime.rc文件中[定义属性](image/sysMaster/runtime-0.png)和[方法](image/sysMaster/runtime-1.png)
  * new和set_state都是方法，前者会把rt的InitState初始化为Init，并且初始化其他属性；后者会把InitState改为Running
  * runloop是一个[死循环](image/sysMaster/runtime-runloop.png)，不断地处理SIGNAL、TIMER、SOCKET等
  * 崩溃快速自愈的实现是由runloop中的is_running来判断，之后会在main中执行reexec

### 1-1 core

core部分实现了很多核心功能，这里拿sctl(systemctl)为例：

```rust
// main函数位于sysmaster/core/sctl/src/main.rs
fn main() {
    let args = parse_args();

    let command_request = match generate_command_request(args) {
        None => {
            eprintln!("This command is currently not supported");
            exit(nix::Error::ENOTSUP as i32);
        }
        Some(v) => v,
    };

    let stream = match UnixStream::connect(PRIVATE_SOCKET) {
        Err(e) => {
            eprintln!("Failed to connect to sysmaster: {}", e);
            exit(e.raw_os_error().unwrap());
        }
        Ok(v) => v,
    };

    let mut client = ProstClientStream::new(stream);

    let data = match client.execute(command_request) {
        Err(e) => {
            eprintln!("Failed to execute the given command: {}", e);
            exit(1);
        }
        Ok(v) => v,
    };

    /* We should always print the error message if the returned error code is not 0. */
    if data.message.is_empty() {
        exit(0);
    }

    if data.error_code == 0 || (data.error_code & ERROR_CODE_MASK_PRINT_STDOUT != 0) {
        /* Don't care if we fail to write the message out. */
        let _ = writeln!(std::io::stdout(), "{}", data.message);
    } else {
        eprintln!("{}", data.message);
    }

    exit((data.error_code & !ERROR_CODE_MASK_PRINT_STDOUT) as i32);
}
```

* parse_args
  * 解析命令行参数
* generate_command_request
  * 生成命令请求
  * 根据解析后的命令行参数生成CommandRequest对象
* connect
  * 连接sysmaster管理器
  * 通过Unix域套接字连接到sysmaster管理器
* ProstClientStream
  * 创建ProstClientStream客户端，并通过execute方法发送命令请求
* 处理返回结果
  * 如果message为空，则表示没有出错；否则做出相应的处理

### 1-2 exts

exts下有**devmaster**(设备初始化和管理)、**fstab**(定义系统上的文件系统如何挂载)、**getty-generator**(生成getty配置或脚本的工具，getty用来初始化终端登录)、**hostname_setup**(配置主机名)、**hwdb**(硬件数据库)、**libudev**(设备管理器的库文件)、**libudev_macro**(支持libudev使用)、**machine-id-setup**(生成机器唯一标识符)、**random_seed**(随机数生成器)、**rc-local-generator**(Linux启动脚本)、**run**(存放运行时状态)、**switch_root**(与切换根文件系统相关)和**sysmonitor**(监控系统状态)

这里以random_seed为例：

```rust
// main函数位于sysmaster/exts/random_seed/src/main.rs
fn main() {
    log::init_log_to_console_syslog("random-seed", log::Level::Debug);
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        log::error!("{}", "This program requires one argument.");
        process::exit(1);
    }

    unsafe {
        libc::umask(0o022);
    }

    if let Err(str) = run(&args[1]) {
        log::error!("{}", str);
        process::exit(1);
    }

    process::exit(0);
}
```

* init_log_to_console_syslog
  * 初始化日志
  * 将日志输出到控制台和系统日志，标签为random_seed，日志级别为Debug，所有级别大于等于Debug的日志消息都被记录
* collect
  * 收集命令行参数
* args.len
  * 检查参数数量，必须为2
* umask
  * 设置文件权限掩码
  * 0o022/新创建的文件权限是-rw-r--r--/目录是drwxr-xr-x
* run
  * 当参数为load时，函数会读取随机种子文件的内容，并可能将部分内容写入/dev/urandom
  * 当参数为save时，函数会生成新的随机数据，并将其保存到随机种子文件中
