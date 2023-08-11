# slimt

**slimt** (_slɪm tiː_) is an inference frontend for
[tiny11](https://github.com/browsermt/students/tree/master/deen/ende.student.tiny11)
[models](https://github.com/browsermt/students) trained as part of the
[Bergamot project](https://browser.mt/).

[bergamot-translator](https://github.com/browsermt/bergamot-translator/) builds
on top of [marian-dev](https://github.com/marian-nmt/marian-dev) and uses the
inference code-path from marian-dev. While marian is a a capable neural network
library with focus on machine translation, all the bells and whistles that come
with it are not necessary to run inference on client-machines (e.g: autograd,
multiple sequence-to-sequence architecture support, beam-search). For some use
cases like an input-method engine doing translation (see
[lemonade](https://github.com/jerinphilip/lemonade)). Single-thread operation
existing along with other processes on the system suffices. This is the
motivation for this transplant repository. There's not much novel here except
easiness to wield. This repository is simply just the _tiny11_ part of marian.
Code is reused where possible.

This effort is inspired by contemporary efforts like
[ggerganov/ggml](https://github.com/ggerganov/ggml) and
[karpathy/llama2](https://github.com/karpathy/llama2.c). tiny11 models roughly
follow the [transformer architecture](https://arxiv.org/abs/1706.03762), with
[Simpler Simple Recurrent Units](https://aclanthology.org/D19-5632/) (SSRU) in
the decoder. The same models are used in Mozilla Firefox's [offline translation
addon](https://addons.mozilla.org/en-US/firefox/addon/firefox-translations/).

The large-list of dependencies from bergamot-translator have currently been
reduced to:

* For `int8_t` matrix-multiply [intgemm](https://github.com/kpu/intgemm) (`x86_64`) or
  [ruy](https://github.com/google/ruy) (`aarch64`, planned).
* For vocabulary - [sentencepiece](https://github.com/browsermt/sentencepiece). 
* For `sgemm` - Whatever BLAS provider is found via CMake. 
* OpenMP is used in `layer_norm`, and is pending removal.
* CLI11 (only a dependency for cmdline) 

Source code is made public where basic functionality (text-translation) works.
Parity in features and speed with marian and bergamot-translator (where
relevant) is a work-in-progress. Contributions are welcome and appreciated.

## Getting started


```bash
# Configure
cmake -B build -S $PWD              \ 
   -DCMAKE_BUILD_TYPE=Release 

# Build
cmake --build build --target all --parallel 4
```

Successful build generate two executables `slimt` and `slimt_test` for
command-line usage and testing respectively.

```bash
build/bin/slimt                     \
    --model </path/to/model>        \
    --vocab </path/to/vocab>        \
    --shortlist </path/to/shortlist>

build/slimt_test <test-name>
```

To fetch a few models to start-development, use the python package:

```bash
# Install `bergamot` CLI via pip.
python3 -m pip install bergamot -f  https://github.com/jerinphilip/bergamot-translator/releases/expanded_assets/latest

# Download en-de-tiny and de-en-tiny models.
bergamot download -m en-de-tiny
bergamot download -m de-en-tiny
```

The following should work assuming a linux system, after:

```bash
export KLIME_DEBUG=1 # Enables printing few array contents
export KLIME_EPS=1e-6

build/slimt-test integration
```
