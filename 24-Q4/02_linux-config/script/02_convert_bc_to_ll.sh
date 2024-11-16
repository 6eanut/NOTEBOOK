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
    # 获取文件的基本路径（去掉扩展名）
    basefile="${file%.bc}"
    # 构建目标文件路径
    llfile="${basefile}.ll"
    # 使用 llvm-dis 将 .bc 文件转换为 .ll 文件
    llvm-dis "$file" -o "$llfile"
    echo "Converted $file to $llfile"
done
