#!/bin/bash

# 检查是否提供了一个参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_directory>"
    exit 1
fi

# 从命令行参数获取源目录，并确保结尾有斜杠
src_dir="$1"
src_dir=${src_dir%/}/  # 如果结尾没有斜杠，则添加一个

# 初始化总基本块计数器
total_blocks=0

# 使用 find 命令从 src_dir 开始递归查找所有 .dot 文件
while IFS= read -r -d '' file; do
  if [ -f "$file" ]; then
    # 获取相对路径
    rel_file="${file#$src_dir}"

    # 使用 grep 命令查找所有包含 [shape=record,color= 的行
    num_blocks=$(grep -c '\[shape=record,color=' "$file")

    # 打印当前文件的基本块数量
    echo "File: $rel_file, Number of basic blocks: $num_blocks"

    # 累加到总基本块计数器
    total_blocks=$((total_blocks + num_blocks))
  fi
done < <(find "$src_dir" -type f -name "*.dot" -print0)

# 打印总的基本块数量
echo "Total number of basic blocks in all .dot files: $total_blocks"
