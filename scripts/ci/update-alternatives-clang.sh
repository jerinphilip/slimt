# !/bin/bash
#
# Taken from comment inside the following gist
# https://gist.github.com/junkdog/70231d6953592cd6f27def59fe19e50d?permalink_comment_id=4336074#gistcomment-4336074
# add clang ${version} to Ubuntu

set -eo pipefail

update_alternatives() {
  local version=${1}
  local priority=${2}
  local master=${3}
  local slaves=${4}
  local path=${5}
  local cmdln

  cmdln="--verbose --install ${path}${master} ${master} ${path}${master}-${version} ${priority}"
  for slave in ${slaves}; do
    cmdln="${cmdln} --slave ${path}${slave} ${slave} ${path}${slave}-${version}"
  done
  sudo update-alternatives ${cmdln}
  sudo update-alternatives --set ${master} ${path}${master}-${version}
}

update_alternatives_master() {
  local version=${1}
  local priority=${2}
  local packages=${3}
  local path=${4}
  local cmdln
  for package in ${packages}; do
    cmdln="--verbose --install ${path}${package} ${package} ${path}${package}-${version} ${priority}"
    sudo update-alternatives ${cmdln}
    sudo update-alternatives --set ${package} ${path}${package}-${version}
  done
}

if [[ ${#} -ne 2 ]]; then
  echo usage: "${0}" clang_version priority
  exit 1
fi

version=${1}
priority=${2}
path="/usr/bin/"

sudo apt update

# download and launch the setup script
wget https://apt.llvm.org/llvm.sh
# sudo bash llvm.sh ${version}
sudo bash llvm.sh ${version} all

master="clang"
slaves="asan_symbolize bugpoint clang-cpp clangd count dsymutil FileCheck ld64.lld ld.lld llc lld lldb lldb-argdumper lldb-instr lldb-server lldb-vscode lld-link lli lli-child-target not obj2yaml opt sanstats split-file UnicodeNameMappingGenerator verify-uselis
torder wasm-ld yaml2obj yaml-bench"
update_alternatives "${version}" "${priority}" "${master}" "${slaves}" "${path}"

packages="clang++ clang-tidy run-clang-tidy"
update_alternatives_master "${version}" "${priority}" "${packages}" "${path}"

# configure with update-alternatives
master="llvm-config"
slaves="llvm-addr2line llvm-ar llvm-as llvm-bcanalyzer llvm-bitcode-strip llvm-cat llvm-cfi-verify llvm-cov llvm-c-test llvm-cvtres llvm-cxxdump llvm-cxxfilt llvm-cxxmap llvm-debuginfo-analyzer llvm-debuginfod llvm-debuginfod-find llvm-diff llvm-dis llvm-dlltool llvm-d
warfdump llvm-dwarfutil llvm-dwp llvm-exegesis llvm-extract llvm-gsymutil llvm-ifs llvm-install-name-tool llvm-jitlink llvm-jitlink-executor llvm-lib llvm-libtool-darwin llvm-link llvm-lipo llvm-lto llvm-lto2 llvm-mc llvm-mca llvm-ml llvm-modextract llvm-mt llvm-nm llv
m-objcopy llvm-objdump llvm-opt-report llvm-otool llvm-pdbutil llvm-PerfectShuffle llvm-profdata llvm-profgen llvm-ranlib llvm-rc llvm-readelf llvm-readobj llvm-reduce llvm-remark-size-diff llvm-remarkutil llvm-rtdyld llvm-sim llvm-size llvm-split llvm-stress llvm-stri
ngs llvm-strip llvm-symbolizer llvm-tapi-diff llvm-tblgen llvm-tli-checker llvm-undname llvm-windres llvm-xray"

update_alternatives "${version}" "${priority}" "${master}" "${slaves}" "${path}"
