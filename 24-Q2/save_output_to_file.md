# Save the output of the command to a file

When running models in tensorflow, it is necessary to save training/inference results. It is more convenient to save the output results to a file, and then customize the name of the file, so that it is particularly convenient in the later viewing process.

**How to use:**

```
command > output_name.txt 2>&1
```

* `>` is the redirect operator, which redirects the standard output (stdout) of the command to the specified file. If the file already exists, this overwrites the original content; If the file does not exist, it will create a new file.
* `2>&1`  is very important because it means to redirect the standard error (stderr, number 2) to the same place as the standard output (stdout, number 1). In other words, this saves all output of the command, whether normal output or error messages, to the output_name.txt file.
