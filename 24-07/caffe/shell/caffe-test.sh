#!/bin/bash

set -e

# 文件列表
output_files=("temp_output.txt" "output-checkout.txt" "output.txt")

# 遍历文件列表
for file in "${output_files[@]}"; do
    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 清空文件
        > "$file"
        echo "Cleared $file"
    else
        echo "$file does not exist."
    fi
done

# 获取所有文件名并存储在一个数组中
files=($(ls))

# 遍历从00到49的文件名
for i in {0..49}
do
    # 构建文件名模式
    pattern=$(printf "%02d-" $i)

    # 使用模式从数组中查找文件名
    for f in "${files[@]}"
    do
        if [[ $f == $pattern* ]]; then
            file=$f
            break
        fi
    done

    # 检查是否找到了匹配的文件
    if [ -z "$file" ]; then
        echo "File not found: ${pattern}*"
    else
        # 执行caffe time命令并将最后10行输出追加到输出文件
        echo ""$file"正在测试中"
        /home/caffe-test/perf_information_get.sh "caffe time -model "$file"" "/home/caffe-test/perf"  &> temp_output.txt
        tail -n 13 temp_output.txt > output.txt
        echo "$file" > output-check.txt
        tail -n 13 temp_output.txt > output-checkout.txt 
        rm temp_output.txt
        echo ""$file"测试完成"
    fi
done
