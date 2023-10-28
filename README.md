# slimt

**slimt** (_slɪm tiː_) is an inference frontend for
[tiny](https://github.com/browsermt/students/tree/master/deen/ende.student.tiny11)
[models](https://github.com/browsermt/students) trained as part of the
[Bergamot project](https://browser.mt/).

[bergamot-translator](https://github.com/browsermt/bergamot-translator/) builds
on top of [marian-dev](https://github.com/marian-nmt/marian-dev) and uses the
inference code-path from marian-dev. While marian is a a capable neural network
library with focus on machine translation, all the bells and whistles that come
with it are not necessary to run inference on client-machines (e.g: autograd,
multiple sequence-to-sequence architecture support, beam-search). For some use
cases like an input-method engine doing translation (see
[lemonade](https://github.com/jerinphilip/lemonade)) - single-thread operation
existing along with other processes on the system suffices. This is the
motivation for this transplant repository. There's not much novel here except
easiness to wield. This repository is simply just the _tiny_ part of marian.
Code is reused where possible.

This effort is inspired by contemporary efforts like
[ggerganov/ggml](https://github.com/ggerganov/ggml) and
[karpathy/llama2](https://github.com/karpathy/llama2.c). _tiny_ models roughly
follow the [transformer architecture](https://arxiv.org/abs/1706.03762), with
[Simpler Simple Recurrent Units](https://aclanthology.org/D19-5632/) (SSRU) in
the decoder. The same models are used in Mozilla Firefox's [offline translation
addon](https://addons.mozilla.org/en-US/firefox/addon/firefox-translations/).

Both `tiny` and `base` models have 6 encoder-layers and 2 decoder-layers, and
for most existing pairs a vocabulary size of 32000 (with tied embeddings). The
following table briefly summarizes some architectural differences between
`tiny` and `base` models:

| Variant | emb | ffn  | params | f32   | i8   |
| ------- | --- | ---  | ------ | ----- | ---- |
| `base`  | 512 | 2048 | 39.0M  | 149MB | 38MB |
| `tiny`  | 256 | 1536 | 15.7M  | 61MB  | 17MB |

The `i8` models, quantized to 8-bit and as small as 17MB is used to provide
translation for Mozilla Firefox's offline translation addon, among other
things.

More information on the models are described in the following papers:

* [From Research to Production and Back: Ludicrously Fast Neural Machine Translation](https://aclanthology.org/D19-5632)
* [Edinburgh’s Submissions to the 2020 Machine Translation Efficiency Task](https://aclanthology.org/2020.ngt-1.26/)


The large-list of dependencies from bergamot-translator have currently been
reduced to:

* For `int8_t` matrix-multiply [intgemm](https://github.com/kpu/intgemm)
  (`x86_64`) or [ruy](https://github.com/google/ruy) (`aarch64`) or
  [xsimd](https://github.com/xtensor-stack/xsimd) via
  [gemmology](https://github.com/mozilla/gemmology).
* For vocabulary - [sentencepiece](https://github.com/browsermt/sentencepiece). 
* For sentence-splitting using regular-expressions
  [PCRE2](https://github.com/PCRE2Project/pcre2).
* For `sgemm` - Whatever BLAS provider is found via CMake (openblas,
  intel-oneapimkl, cblas).  Feel free to provide
  [hints](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors). 
* [CLI11](https://github.com/CLIUtils/CLI11/) (only a dependency for cmdline) 

Source code is made public where basic functionality (text-translation) works
for English-German tiny models. Parity in features and speed with marian and
bergamot-translator (where relevant) is a work-in-progress. Eventual support for
`base` models are planned. Contributions are welcome and appreciated.


## Getting started

Clone with submodules.

```
git clone --recursive https://github.com/jerinphilip/slimt.git
```

Configure and build. `slimt` is still experimenting with CMake and
dependencies. The following should work at the moment:


```bash

# Configure intgemm
cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release -DWITH_INTGEMM=ON 
# Configure ruy instead of intgemm
cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release -DWITH_RUY=ON

# Build
cmake --build build --target all --parallel 4
```

Successful build generate two executables `slimt-cli` and `slimt-test` for
command-line usage and testing respectively.

```bash
build/bin/slimt-cli                           \
    --root <path/to/folder>                   \
    --model </relative/path/to/model>         \
    --vocabulary </relative/path/to/vocab>    \
    --shortlist </relative/path/to/shortlist>

build/slimt-test <test-name>
```

### Distribution

There is a build-path being prepared towards packaging on Linux. To use this,
configure with the following args:

```bash
# Configure to use xsimd via gemmology
ARGS=(
    # Use gemmology
    -DWITH_GEMMOLOGY=ON               

    # -DUSE_AVX512 -DUSE_SSSE3 ... -DUSE_NEON also available.
    -DUSE_AVX2=ON                          

    # Use sentencepiece installed via system.
    -DUSE_BUILTIN_SENTENCEPIECE=OFF        

    # Exports slimtConfig.cmake (cmake) and slimt.pc.in (pkg-config)
    -DSLIMT_PACKAGE=ON 

    -DCMAKE_INSTALL_PREFIX=/path/to/prefix
)

cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release "${ARGS[@]}"
cmake --build build --target all

# May require if prefix is writable only by root.
cmake --build build --target install 
```

The above run expects the packages `sentencepiece`, `xsimd` and a BLAS provider
to come from the system's package manager. Examples of this in distributions
include:

```bash
# Debian based systems
sudo apt-get install -y libxsimd-dev libsentencepiece-dev libopenblas-dev

# ArchLinux
pacman -S openblas xsimd
yay -S sentencepiece-git
```

This is still very much a work in progress, towards being able to make
[lemonade](https://github.com/jerinphilip/lemonade) available in distributions.
Help is much appreciated here, please get in touch if you can help here.

### Python

Python bindings to the C++ code are available.  Python bindings provide a layer
to download models and use-them via command line entrypoint `slimt` (the core
slimt library only has the inference code).

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install wheel
python3 setup.py bdist_wheel
python3 -m pip install dist/<wheel-name>.whl

# Download en-de-tiny and de-en-tiny models.
slimt download -m en-de-tiny
slimt download -m de-en-tiny
```

You may pass customizing cmake-variables via `CMAKE_ARGS` environment variable.

```bash
CMAKE_ARGS='-D...' python3 setup.py bdist_wheel
```
