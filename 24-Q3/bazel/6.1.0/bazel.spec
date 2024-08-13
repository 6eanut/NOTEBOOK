%define _disable_source_fetch 0

Name:           bazel
Version:        6.1.0
Release:        2
Summary:        Correct, reproducible, and fast builds for everyone.
License:        Apache License 2.0
URL:            http://bazel.io/
Source0:        bazel-%{version}-dist.zip
Source1:        abseil-cpp-riscv.patch
Patch0:         01-fix-invalid-jni_md-select.patch
Patch1:         02-use-riscv64-jdk.patch
Patch2:         03-set-default-nojdk.patch
Patch3:         linux-bazel-path-from-getauxval.patch
Patch4:         04-riscv-distdir_deps.patch
Patch5:		05-bazel610.patch
# for folks with 'bazel' v1 package installed
Conflicts:      bazel
Conflicts:      bazel2

BuildRequires:  java-11-openjdk-devel zlib-devel findutils gcc-c++ which unzip zip python3
BuildRequires:  pkgconfig(bash-completion)

Requires:       java-11-openjdk-devel

%define bashcompdir %(pkg-config --variable=completionsdir bash-completion 2>/dev/null)
%global debug_package %{nil}
%define __os_install_post %{nil}

%description
Correct, reproducible, and fast builds for everyone.

%prep
%setup -q -c -n bazel-%{version}
#%patch0 -p1
#%patch1 -p1
#%patch2 -p1
#%patch3 -p1
%ifarch riscv64
#%patch4 -p1
#%patch5 -p1
mkdir third_party/abseil-cpp
cp %{SOURCE1} third_party/abseil-cpp
%endif

%build
find . -type f -regextype posix-extended -iregex '.*(sh|txt|py|_stub|stub_.*|bazel|get_workspace_status|protobuf_support|_so)' -exec %{__sed} -i -e '1s|^#!/usr/bin/env python$|#!/usr/bin/env python3|' "{}" \;
export EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS} --python_path=/usr/bin/python3"

# horrible of horribles, just to have `python` in the PATH
%{__mkdir_p} ./bin-hack
%{__ln_s} /usr/bin/python3 ./bin-hack/python
export PATH=$(pwd)/bin-hack:$PATH

%ifarch aarch64
export EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS} --nokeep_state_after_build --notrack_incremental_state --nokeep_state_after_build"
%else
%endif

%ifarch aarch64 %arm riscv64
export BAZEL_JAVAC_OPTS="-J-Xmx2g -J-Xms200m"
%endif

%ifarch s390x
# increase heap size to addess s390x build failures
export BAZEL_JAVAC_OPTS="-J-Xmx4g -J-Xms512m"
%else
%endif

# loose epoch from their release date
export SOURCE_DATE_EPOCH="$(date -d $(head -1 CHANGELOG.md | %{__grep} -Eo '\b[[:digit:]]{4}-[[:digit:]]{2}-[[:digit:]]{2}\b' ) +%s)"
export EMBED_LABEL="%{version}"

# for debugging's sake
which g++
g++ --version

export TMPDIR=%{_tmppath}
export CC=gcc
export CXX=g++
export EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS} --sandbox_debug --tool_java_runtime_version=local_jdk --verbose_failures --subcommands --explain=build.log --show_result=2147483647"
env ./compile.sh
env ./scripts/generate_bash_completion.sh --bazel=output/bazel --output=output/bazel-complete.bash

%install
%{__mkdir_p} %{buildroot}/%{_bindir}
%{__mkdir_p} %{buildroot}/%{bashcompdir}
%{__cp} output/bazel %{buildroot}/%{_bindir}/bazel-real
%{__cp} output/bazel %{buildroot}/%{_bindir}/bazel-%{version}-%{_os}-%{_arch}
%{__cp} ./scripts/packages/bazel.sh %{buildroot}/%{_bindir}/bazel
%{__cp} output/bazel-complete.bash %{buildroot}/%{bashcompdir}/bazel

%clean
%{__rm} -rf %{buildroot}

%files
%defattr(-,root,root)
%attr(0755,root,root) %{_bindir}/bazel
%attr(0755,root,root) %{_bindir}/bazel-real
%attr(0755,root,root) %{_bindir}/bazel-%{version}-%{_os}-%{_arch}
%attr(0755,root,root) %{bashcompdir}/bazel

%changelog
* Tue Jul 09 2024 laokz  <zhangkai@iscas.ac.cn> - 5.3.0-2
- riscv64: fix 'rdcycle' SIGILL of dependency abseil-cpp

* Tue Sep 12 2023 Jincheng Miao  <jincheng.miao@intel.com> - 5.3.0-1
- add bazel-%{version}-%{_os}-%{_arch} to install root path

* Fri Sep 08 2023 Jincheng Miao  <jincheng.miao@intel.com> - 5.3.0-0
- upgrade bazel to 5.3.0 for building TF-2.12.1

* Fri Jun 09 2023 Jingwiw  <wangjingwei@iscas.ac.cn> - 5.1.1-1
- fix riscv64 build error and add qemu user support

* Tue Nov 8 2022 Jincheng Miao <jincheng.miao@intel.com> - 5.1.1-0
- Update version to 5.1.1 for TF-2.10.0 build

* Sat Oct 22 2022 Jincheng Miao <jincheng.miao@intel.com> - 5.0.0-0
- Update version to 5.0.0 for TF-2.9 build

* Thu Jun 22 2022 zhangshaoning <zhangshaoning@uniontech.com> - 4.2.1-0
- update version to 4.2.1

* Mon Aug 9 2021 zhangtao <zhangtao221@huawei.com> - 3.5.0-4
- fix build error with gcc10

* Tue Jul 13 2021 guoxiaoqi <guoxiaoqi2@huawei.com> - 3.5.0-3
- Not strip %{_bindir}/bazel after install

* Mon May 31 2021 baizhonggui <baizhonggui@huawei.com> - 3.5.0-2
- Add gcc-g++ in BuildRequires

* Mon Sep 28 2020 Zhipeng Xie<xiezhipeng1@huawei.com> - 3.5.0-1
- Package init
