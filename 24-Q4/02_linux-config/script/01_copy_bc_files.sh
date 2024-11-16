#!/bin/bash

# 检查是否提供了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# 从命令行参数获取源目录和目标目录
src_dir="$1"
dst_dir="$2"

# 创建目标目录（如果不存在的话）
mkdir -p "$dst_dir"

# 使用find命令查找所有.bc文件并处理
find "$src_dir" -type f -name "*.bc" | while read file; do
    # 计算相对于源目录的相对路径
    relpath=$(realpath --relative-to="$src_dir" "$file")
    # 构建目标文件路径
    dstfile="$dst_dir/$relpath"
    # 确保目标文件所在的目录存在
    mkdir -p "$(dirname "$dstfile")"
    # 复制文件
    cp "$file" "$dstfile"
done
