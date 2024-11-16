#!/bin/bash

# 检查是否提供了一个参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <source_directory>"
    exit 1
fi

# 从命令行参数获取源目录
src_dir="$1"

# 使用find命令查找所有.bc文件并处理
find "$src_dir" -type f -name "*.bc" | while read file; do
    # 获取文件所在目录
    dir=$(dirname "$file")

    # 切换到文件所在目录
    cd "$dir" || { echo "Failed to cd to $dir"; continue; }

    # 获取文件名（不包括路径）
    base=$(basename "$file" .o.bc)
    base01="${base#.}"_dot
    mkdir -p "$base01"
    cd "$base01" || { echo "Failed to cd to $base01"; continue; }

    # 使用 opt 工具生成 .dot 文件
    opt -passes=dot-cfg "$file"
    echo "Generated .dot files for $file at $dir/$base01"
done
