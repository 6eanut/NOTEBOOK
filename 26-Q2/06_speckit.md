# Spec Kit

## 1 环境搭建

### 1-1 临时运行

如果还没有项目目录

```
uvx --from git+https://github.com/github/spec-kit.git specify init blog-system
uvx --from git+ssh://git@github.com/github/spec-kit.git specify init blog-system
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
uvx --from git+ssh://git@github.com/github/spec-kit.git specify init .
```

### 1-2 永久安装

```
pipx install git+https://github.com/github/spec-kit.git
pipx install git+ssh://git@github.com/github/spec-kit.git
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

## 3 Try

需要先配一下opencode，因为spec-kit好像不官方支持claude code，然后运行

```
uvx --from git+https://github.com/github/spec-kit.git specify init 2026fifaworldcup
opencode
```

接下来进入SDD工作流：

```
/speckit.constitution
本项目是 2026 FIFA World Cup 数据可视化前端
核心原则：
1. Static First
   所有数据必须为静态 JSON，不依赖后端
2. Visualization First
   所有信息必须优先图表化，而不是文本堆叠
3. Minimal & Premium UI
   风格参考 Apple / Stripe / Linear
   简约、留白、卡片化设计
4. Mobile First
   所有页面必须适配手机端
5. TypeScript Strict Mode
   禁止 any
6. Component Driven
   UI 必须组件化拆分，不写巨型页面
7. GitHub Pages Deploy Ready
   必须支持纯静态部署
8. Data Integrity
   不允许伪造“种族”等敏感不可验证数据
   只使用可公开验证的数据维度（国家、联赛、年龄等）

/speckit.specify
构建一个 2026 FIFA World Cup 球员数据分析网站
功能包括：
1. 球员年龄分布
2. 球员所属联赛分布（五大联赛 + 其他）
3. 球员俱乐部分布 Top 20
4. 各国家队球员结构分析
5. 各洲球员分布（UEFA / CONMEBOL / CAF / AFC / CONCACAF）
6. 世界杯经验统计（首次 / 多届）
7. 球员身价分布（区间统计）
8. 有趣数据排行（Top N）
视觉要求：
- 卡片式布局
- 数据驱动图表
- 首页 dashboard 风格
- 支持深色模式
- 风格简约现代（Apple / Stripe）
目标：
GitHub Pages 静态部署

/speckit.clarify
/speckit.checklist
/speckit.plan
/speckit.tasks
/speckit.analyze
/speckit.implement
```

最后做出来的效果是：

https://6eanut.github.io/2026FIFAWorldCup/

## 4 Think

整体感觉是：1.很慢；2.效果感觉一般
