# bluespec-lsp：AI实现的Rust项目，为BSV开发者带来更佳体验

## 0 背景

随着AI大模型技术的迅猛发展，RDMA技术所提供的高性能网络已成为AI基础设施的关键组成部分。然而，诞生于上世纪的RDMA技术难以完全适配当今的AI计算场景：

* 现有商用RDMA方案受限于历史兼容性负担，难以快速迭代以满足AI需求；
* RDMA网卡作为闭源黑盒产品，与倡导开源的上层AI大模型软件生态形成显著矛盾。

open-rdma 由琶洲实验室（黄埔）和 Datanlord 协作发起，可以从硬件到软件自由定制和优化。open-rdma 不是一个单独的仓库，而是一组互相配合的子项目：

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=NWE2NjM5ZjQzMWE5NzExZWI3NmRkMjIwOWQ5OWM0ODlfTFlyZFdHQkNoMVB1V0VrUE1XQ3VWcThZMTZhcjM2RjRfVG9rZW46UXFvb2JVSDdwb0Joekp4Q1o0QmM5WmtCblBiXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

bluespec-lsp是open-rdma项目矩阵里的一个子项目。

## 1 为什么需要一个bluespec-lsp？

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=MWZkODdhZGI2Y2JmYmU0ZWVhOTBlNjk1ZjNlZWJmMGZfN1ZKV0tWaWIwSE5EUU1qSlprTWUxazRsY3lVWFBSb0hfVG9rZW46RVZMVmJSSGpSb0tWTlp4aVZyWmNGSDNPbnFnXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

BSV 在学术圈、处理器、网络硬件中，均被广泛开发使用。可以看到，open-rdma 的硬件部分用的是 Bluespec SystemVerilog (BSV)。并且，一些知名开源处理器(CHERI-BERI, Flute等)采用BSV来开发，MIT 著名的计算机体系结构课程(MIT 6.175)所采用的是BSV，商业网络ASIC会用BSV设计高速网络处理器和交换机芯片。

虽然很多项目在用BSV，但是BSV的IDE支持却很少。BSV 官方工具链只有一个命令行编译器 bsc——编译没问题，但没有语法高亮、没有跳转、没有补全、没有 hover。

想象一下：如果一个BSV项目包含几百(千)行，想跳转到某个interface的定义，想知道一个嵌套了3(4)层的#define常量展开后是多少，可能需要人工去grep或自己计算一遍常量值，开发效率很低。

所以，一个bluespec-lsp能够给BSV开发者带来更佳的开发体验。

## 2 Coding Agent/Coding Language Model帮我们实现！

开发bluespec-lsp的motivation有了，接下来是不是该找几个有经验的开发者做架构设计、开发、测试、迭代了？随着AI技术的快速发展以及能力越来越强，一个人加一个Coding Agent/Coding Language Model，用两天的时间就可以做成这件事了。

接下来，我会和大家分享一下整个的开发过程——用了什么工具、如何配置、如何使用、人在中间扮演什么角色。

市面上 AI 编程工具有不少——Cursor、GitHub Copilot、Windsurf、Aider 等等。我选择了Claude Code，有几个原因：

### 2-1 Plan Mode

用 `/plan` 进入规划模式后，AI 不会写代码，而是先深度探索项目代码库，理清现有架构和数据流，然后产出一份完整的实施计划：分几个阶段、每个阶段改哪些文件、有什么风险、依赖关系是什么。计划写完之后，你审阅，批准了才开始写代码。

这个机制的价值怎么强调都不为过。以前用 AI 写代码最怕什么？最怕它一上来就狂写几百行，等你发现方向不对的时候已经晚了。Plan Mode 把这个风险前置消灭了——你在它写代码之前就知道它打算怎么干，不同意的当场改。bluespec-lsp 的模块拆分、tree-sitter C parser 的集成方案设计，这些架构决策都是在 Plan Mode 里敲定的。

### 2-2 Agent Teams

写一个 LSP 涉及多个独立的功能模块——parser、符号表、hover、completion、VS Code 客户端。传统做法是一个一个来，改完一个再改下一个。

Claude Code 的 Agent Teams 可以同时派出多个专业 agent，各自负责一个独立子任务，并行推进。比如语法高亮覆盖不全、符号跳转的边界 case、VSIX 打包脚本优化——这三个任务互不依赖，三个 agent 一起去搞，搞完汇总回来。

关键点是隔离——每个 agent 有独立的上下文窗口，互相不干扰。脑力没有被分摊，而是真的在并行。实际体验就是：三件事情同时推进，而且每个都比单独一个 agent 做三件事更靠谱。

### 2-3 RALPH Loop

把同一个 prompt 反复喂给 Claude，它每次看到自己上一轮的产出（文件改动、git diff、测试结果），继续改进。

`/ralph-loop` 就是这个理念的产品化。你给它一个明确的目标和完成标准（比如"所有 cargo test 通过"），设置最大迭代次数，然后就可以去喝杯咖啡了。它会自己写代码 → 跑测试 → 看结果 → 修正 → 再跑测试，循环到通过为止。

在这个项目中，Ralph Loop 大量用于语法覆盖的补充和测试失败修复——"把这个 .bsv 文件里所有语法节点正确分类"、"修好这 3 个 error recovery 测试失败"——这类目标明确、有客观验证标准（跑测试）的任务，丢给 Ralph Loop 比人盯着改效率高得多。

### 2-4 Code Review

AI 写代码很快，但 AI 写的代码不能直接信。有了代码审查 agent，每次 AI 完成一个 feature，马上用另一个独立的 agent 做 review——检查安全性、逻辑正确性、代码风格、测试覆盖。

这不是表面走流程。review agent 真的会跑 `cargo test` +`cargo build` 来验证。在 bluespec-lsp 的开发过程中，review agent 抓到过不少 AI 自己写出来的问题。而且用 AI 来做 code review 有个天然优势：它不会像人一样在第 40 个文件时审美疲劳。

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=MTM1YjBkNjIzMWE2ZTlkNjYxYjliMzQ5ZTk1ZTRhM2VfTk9YaXBRUFR5WEN2OTB0WnVKYmVrZW5yM2JPVHhNb3BfVG9rZW46TmRXS2I1VUx6b1A4ekd4cnBwQmNVdXdlbm5iXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

这四个合在一起，形成了一套完整的 AI 辅助开发工作流：Plan Mode 出方案 → Agent Teams 并行实现 → Code Review 把关 → Ralph Loop 迭代修复

整个 bluespec-lsp 的开发期，我几乎没有手动写过一行 Rust 代码。我的角色是：定义需求、审方案、审代码、在 VS Code Extension Host 里实际测试、把 bug 丢回去让 AI 修。

## 3 目前有哪些功能？

### 3-1 语法高亮

语法高亮是最基础但也最影响日常体验的功能。

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=N2U5YWY1ODk0ZTE1YTEwMDAwNzAzOWMzMjllN2M3NDNfSHVZREVXV091a0g3OGxIejhHUEJSSEtGakdLNWQyTVJfVG9rZW46WDJDV2JUMUxnb1FaYlh4QnNLR2NBU3RsbkY1XzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

### 3-2 导航跳转

显示当前文件的符号大纲，并支持直接跳转

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=NjA4NzJkYzk1Mjc1ODBhYjc3NmRjNTZkMzU3NDNhNjJfNWNLRjZqbWpsb3c2MERKQm1uVjJ0ZExDRzFxbDJjeGtfVG9rZW46TjQ2VGJGb25Jb1V2elB4TGI2QmN2NGJ2bmpoXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

### 3-3 符号跳转

光标放在一个模块名或函数名上，光标跳转到它的定义处。

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGI4MDU2YzUzYzA4NzkxZWJhMDQ1NmYxNmVhZDY3YjFfWDR4Q3NIUVZ1bmVMMEZ4aUdoNGVvV0tKVXl5Y1UzRHVfVG9rZW46RkZsZWJ6cDFhb0hCUzB4S1lWSmM3YlhmblBiXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

### 3-4 常量自动展开

光标悬浮在常量上时，可以显示其定义过程及具体值

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=OGJlYTgyY2Q3ZDE0YWU1N2EyNGE0NTUwZmY2ZDgwYzZfMDdRaWVjaE9HdWEyRW1DaUYyYkZic0xOWXRLTWVFVTlfVG9rZW46TlQ0NWI2cFFCb1Vkb094dm95Z2NYV0NjblcxXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

## 4 如何参与进来？

bluespec-lsp 是一个社区驱动的开源项目，欢迎任何人参与贡献。更重要的是，这个项目的贡献方式本身就是 AI-native 的——你不需要精通 Rust，不需要理解 LSP 协议，甚至不需要熟悉 BSV 语法。你只需要：

* **有一个 Claude Code 环境**

如果你还没有，可以先看看** **[Claude Code 的官方文档](https://docs.anthropic.com/en/docs/claude-code/overview) 完成安装和配置。

* **Clone 仓库，跑一遍 `/init`**

```Python
git clone https://github.com/open-rdma/bluespec-lsp.git
cd bluespec-lsp
claude
```

然后在 Claude Code 里输入 `/init`。AI 会扫描整个项目，生成 `CLAUDE.md`，确保它对项目的架构、构建命令、模块间的关系有充分理解。

* **挑一个 Issue，描述给 AI**

从[Good First Issue](https://github.com/open-rdma/bluespec-lsp/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) 列表里找一个感兴趣的。把 Issue 的标题和描述直接告诉 AI，加上一句"读取这个 issue 的详细内容，然后实现它"——AI 会自己去 GitHub 拉取 Issue 内容和相关代码。

* **AI 写代码，你来审**

AI 会在 Plan Mode 里先出方案，你审过之后才开始写。代码写完后运行 Code Review agent 检查，你再在 VS Code Extension Host 里实际测试功能。通过了就提交 PR。

整个流程不需要你具备深层 Rust 知识。你花时间的是判断"这个方案对不对"和"交互体验好不好"，而不是纠结构建为什么报错。

下面是三个非常适合入门 Good First Issue，如果你对其中任何一个感兴趣，现在就开搞：

### 4-1 Github Actions构建多平台VSIX

 **当前状态** ：GitHub Actions 的 release 流程只构建 Windows x86_64 的 VSIX 包。你在 macOS 或 Linux 上想用这个扩展，只能自己从源码编译。

 **目标** ：把 release workflow 扩展为多平台矩阵构建(Windows/Linux/MacOS + X86/Arm64)

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmI3ZTk5ZWJhNmJjZDg5MmZjZjI4ZWRlZjgzYzZlY2RfRmFXMGpLSW1ySVdzN016SFkzcGFacnJqbWVIQWlNNXZfVG9rZW46SXZUQ2JhcTBib3FxR1V4TkNuU2NQSWt4bmFiXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

 **涉及内容** ：修改 .github/workflows/release.yml，把 runs-on: windows-latest 的单平台构建改成 matrix strategy，在 6 个平台上分别编译 Rust server 和打包 VSIX，最后合并到一个 GitHub Release 里。Rust 交叉编译、setup-node 多平台配置、vsce package 打包——这些都是现成的工具链，主要工作是编排 CI 流程。

### 4-2 支持更多Type Function

 **当前状态** ：常量展开系统支持 8 种类型函数（TAdd、TSub、TMul、TDiv、TLog、TExp、TMax、TMin）。

 **目标** ：扩展到 14 种，新增 6 种位运算和移位函数：

![](https://fcnej5cjit8f.feishu.cn/space/api/box/stream/download/asynccode/?code=MjJmYTRhNzFiNGQ2MDM5OWQ1NmZlZjc0ZmU4YjI4NjVfUlRWUXFoM25URkZZakZUUldQNzBCNnp0TDNIVTRKOEhfVG9rZW46Uzk2RWJTN0g4b1EwWll4NnF6VmNhYmkzbnJkXzE3NzczNjI2MzA6MTc3NzM2NjIzMF9WNA)

 **涉及代码** ：

1. bsv-language-server/src/constant_expansion/types.rs：在 TypeFunction enum 里新增 6 个 variant，在 from_name()里加对应匹配，在 evaluate() 里加运算逻辑（分别对应 Rust 的 `&`、`|`、`^`、`!`、`<<`、`>>` 运算符），在 name() 里返回函数名。
2. bsv-language-server/src/constant_expansion/mod.rs：加测试用例，覆盖每种新函数的基础运算和嵌套展开场景。

如果你参考现有的 TAdd/TSub 的实现模式，加一个新函数大约只需要在每个方法里多 3-5 行代码。这是一个非常好的"通过模仿已有模式来学习项目结构"的切入点。

### 4-3 注释语言统一

 **当前状态** ：代码中的注释中文和英文混用。比如 server.rs 里有 "解析文档并更新符号表"，也有 "Check if this word is a defined constant"。

 **目标** ：将代码库中所有中文注释翻译为英文。

 **涉及内容** ：遍历 bsv-language-server 的 Rust 源码（src/ 目录）、TypeScript 客户端代码（client/ 目录）、以及 tree-sitter-bsv 的 JS/C 文件。不需要改任何逻辑，只是逐文件把中文注释转成对应的英文表述。

## 5 总结

从"BSV 需要好用的 LSP"这个想法出发，到一个人 + Claude Code 几天时间做出一个可用版本，再到把项目开源、邀请大家一起参与，整个过程可能就是AI时代软件开发的一个例子。

如果你觉得这次分享对你有点帮助或者你也想尝试用AI去做一些开源贡献，欢迎参与到 open-rdma 和 bluespec-lsp。点个star、上手体验一下、给我们反馈(提issue)、贡献代码（提pr）～
