# How to Get SHA256

所要计算sha256的文件url为 `file_url`；文件名为 `file_name`.

```
curl -L -o file_name file_url
shasum -a 256 file_name
```
