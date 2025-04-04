# Save the output of the command to a file

When running models in tensorflow, it is necessary to save training/inference results. It is more convenient to save the output results to a file, and then customize the name of the file, so that it is particularly convenient in the later viewing process.

**How to use:**

```
command > output_name.txt 2>&1
```

* `>` is the redirect operator, which redirects the standard output (stdout) of the command to the specified file. If the file already exists, this overwrites the original content; If the file does not exist, it will create a new file.
* `2>&1`  is very important because it means to redirect the standard error (stderr, number 2) to the same place as the standard output (stdout, number 1). In other words, this saves all output of the command, whether normal output or error messages, to the output_name.txt file.

---

## tee工具(20250404)

tee是Linux系统中的一个工具，名称来源于T型管道，功能是同时将输入数据写入终端和文件，像字母T一样分流数据。

用法：

```
command | tee output.log	# 将 command 的输出同时保存到 output.log 并打印到终端
command | tee a output.log	# 追加到文件（不覆盖原有内容）
command 2>&1 | tee output.log	# 将标准错误（stderr）重定向到标准输出（stdout），确保错误信息也被捕获
```
