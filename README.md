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


The large-list of dependencies from bergamot-translator have currently been
reduced to:

* For `int8_t` matrix-multiply [intgemm](https://github.com/kpu/intgemm) (`x86_64`) or
  [ruy](https://github.com/google/ruy) (`aarch64`).
* For vocabulary - [sentencepiece](https://github.com/browsermt/sentencepiece). 
* For `sgemm` - Whatever BLAS provider is found via CMake. 
* CLI11 (only a dependency for cmdline) 

Source code is made public where basic functionality (text-translation) works
for English-German tiny models. Parity in features and speed with marian and
bergamot-translator (where relevant) is a work-in-progress. Eventual support for
`base` models are planned. Contributions are welcome and appreciated.

Both `tiny` and `base` models have 6 encoders and 2 decoders, and for most
existing pairs a vocabulary size of 32000 (with tied embeddings). The following
table briefly summarizes some architectural differences between `tiny` and
`base` models:

| Variant | emb | ffn  | params | f32/i8   | 
| ------- | --- | ---  | ------ | -------- | 
| Base    | 512 | 2048 | 39.0M  | 149/38MB |
| Tiny    | 256 | 1536 | 15.7M  |  61/17MB |

More information on the models are described in the following papers:

* [From Research to Production and Back: Ludicrously Fast Neural Machine Translation](https://aclanthology.org/D19-5632)
* [Edinburgh’s Submissions to the 2020 Machine Translation Efficiency Task](https://aclanthology.org/2020.ngt-1.26/)


## Getting started

Clone with submodules.

```
git clone --recursive https://github.com/jerinphilip/slimt.git
```

Configure and build.

```bash
# Configure
cmake -B build -S $PWD -DCMAKE_BUILD_TYPE=Release 

# Build
cmake --build build --target all --parallel 4
```

Successful build generate two executables `slimt` and `slimt_test` for
command-line usage and testing respectively.

```bash
build/bin/slimt                               \
    --root <path/to/folder>                   \
    --model </relative/path/to/model>         \
    --vocab </relative/path/to/vocab>         \
    --shortlist </relative/path/to/shortlist>

build/slimt_test <test-name>
```

To procure models, use the existing Python CLI.

```bash
# Install `bergamot` CLI via pip.
python3 -m pip install bergamot -f  https://github.com/jerinphilip/bergamot-translator/releases/expanded_assets/latest

# Download en-de-tiny and de-en-tiny models.
bergamot download -m en-de-tiny
bergamot download -m de-en-tiny
```
