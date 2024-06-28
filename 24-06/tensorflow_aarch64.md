# Tensorflow on Aarch64

Setting up a tensorflow environment includes installing tensorflow, tensorflow-text, and tensorflow-models.

## tensorflow

* **github:**https://github.com/tensorflow/tensorflow
* **version:**2.15.0
* **method:**pip install tensorflow==2.15.0

Run `pip show tensorflow`, and if the version number appears, the tensorflow installation is complete.

## tensorflow-text

* **github:**https://github.com/tensorflow/text
* **version:**2.15.0
* **method:**build from source

*step1 : bazel*

Pypi does not have tensorflow-text for the aarch64 architecture, so it needs to be compiled and installed from source. Github explains how to compile and install from source, but bazel is used to build the wheel.
Bazel is available as a release on github and can be downloaded directly. The error message in the construction of the wheel of text will show which version of bazel is required, and you can download the corresponding version of bazel.
Tensorflow-text 2.15.0 requires bazel version 6.1.0.

```
(venv00) [tf-test@jiakai-openeuler-01 file]$ wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-linux-arm64
(venv00) [tf-test@jiakai-openeuler-01 file]$ chmod +x bazel-6.1.0-linux-arm64
(venv00) [tf-test@jiakai-openeuler-01 file]$ mv bazel-6.1.0-linux-arm64 bazel
(venv00) [tf-test@jiakai-openeuler-01 file]$ ./bazel
...
(venv00) [tf-test@jiakai-openeuler-01 file]$ ls
bazel  text
(venv00) [tf-test@jiakai-openeuler-01 file]$ pwd
/home/tf-test/file
(venv00) [tf-test@jiakai-openeuler-01 file]$ export PATH="$PATH:/home/tf-test/file"
```

Run `bazel --version`, and if the version number appears, the bazel installation is complete.

*step2 : text*

```
git clone https://github.com/tensorflow/text
cd text
git checkout 2.15
./oss_scripts/run_build.sh
pip install tensorflow_text-2.15.0-cp311-cp311-linux_aarch64.whl
```

Run `pip show tensorflow-text`, and if the version number appears, the text installation is complete.

## tensorflow-models

* **github:**https://github.com/tensorflow/models
* **version:**2.15.0
* **method:**pip install tf-models-official==2.15.0

Run `pip install tf-models-official`, and if the version number appears, the models installation is complete.

## Summary

At the time of writing this article, I've already run some models with tensorflow. However, recently when running projects in models, there are still some problems. My personal guess is version incompatibility. So I set up the venv virtual environment, and found that tensorflow and models have wheel for aarch64, but text does not. Combining the existing tensorflow and models wheel version number, 2.15.0 was adopted. Because there is no text wheel, it is built from the source code.

The actual build process certainly encountered more problems than the ones documented in this article, such as hdf5 build failures, etc. But since there are corresponding workarounds available online, they are not documented here.

In short, the most important thing to debug a problem is to understand the error message and then solve it.
