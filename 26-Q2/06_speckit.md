# Spec Kit

## 1 环境搭建

### 1-1 临时运行

如果还没有项目目录

```
uvx --from git+https://github.com/github/spec-kit.git specify init blog-system
```

会创建

```
blog-system/
├── specs/
├── .specify/
├── ...
```

如果已经有了项目目录

```
uvx --from git+https://github.com/github/spec-kit.git specify init .
```

### 1-2 永久安装

```
pipx install git+https://github.com/github/spec-kit.git
```

安装完毕后可以直接

```
specify init .
```

## 2 完整工作流

### 2-1 constitution

```
/speckit.constitution
```

做这个项目时必须遵守什么原则

比如

```
本项目采用 Library First（库优先）
所有功能先做成可复用库
严格测试驱动开发(TDD)
优先使用函数式编程
```

### 2-2 Specify

```
/speckit.specify
```

做什么

比如

```
做一个照片管理软件
可以创建相册
相册按日期分组
相册可以拖拽排序
相册不能嵌套
相册内照片用网格形式展示
```

### 2-3 Clarify

```
/sepckit.clarify
```

人写需求时会遗漏很多东西，AI可以通过提问来完善

比如

```
专门检查：
安全问题
性能问题
```

### 2-4 Checklist

```
/speckit.checklist
```

检查需求是否完整

### 2-5 Plan

```
/speckit.plan
```

讨论技术方案

比如：

```
前端：
Vite
界面：
HTML + CSS + JS
数据库：
SQLite
所有图片只存本地
```

### 2-6 Tasks

```
/speckit.tasks
```

把方案拆成开发任务

### 2-7 Analyze

```
/speckit.analyze
```

检查需求-方案-任务的一致性

### 2-8 Implement

```
/speckit.implement
```

开始编码

## 3 Example

这是社区给的一个example

```
/speckit.constitution Taskify is a "Security-First" application. All user inputs must be validated. We use a microservices architecture. Code must be fully documented.

```
