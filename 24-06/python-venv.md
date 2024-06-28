# Python-venv

venv:**V**irtual **ENV**ironment

## 创建venv

python -m venv /home/venv/venv-test

```
[venv@jiakai-openeuler-01 venv-test]$ ls
bin  include  lib  lib64  pyvenv.cfg
[venv@jiakai-openeuler-01 venv-test]$ pwd
/home/venv/venv-test

```

## 切换venv

source /home/venv/venv-test/bin/activate

```
[venv@jiakai-openeuler-01 ~]$ source /home/venv/venv-test/bin/activate
(venv-test) [venv@jiakai-openeuler-01 ~]$
```

## 退出venv

deactivate

```
(venv-test) [venv@jiakai-openeuler-01 ~]$ deactivate
[venv@jiakai-openeuler-01 ~]$

```

## 感想

像比venv功能更强大的有比较知名的anaconda、conda、pyenv等等，他们不仅可以创建虚拟环境，还可以指定虚拟环境的python版本，不像venv，只能用base环境的python版本。但是个人感觉如果不需要切换python版本的话，用venv就够了。

比如在一个venv下，可以从零开始搭建环境，很方便，心里很清透。搭建失败了或者不用了，可以直接删掉。

最近在鲲鹏920上先后测试了caffe和tensorflow，我的做法就是用root用户先后创建了两个用户，分别用了测试caffe和tensorflow，这样也行，但是感觉这种抽象层次太高了，完全可以在一个用户下，抽象出两个venv，然后在不同的venv里面做测试，感觉这样更舒服一点。

但是，如果像之前移植opengauss的项目，那最好还是用anaconda，因为它可以切换python版本。所以到底用不用anaconda，用不用venv，还是要看具体情况。

接触项目越多，越发现版本的重要性，还是要先弄清楚版本，然后会更方便。
