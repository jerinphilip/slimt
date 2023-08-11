#!/bin/bash

BROWSERMT=$HOME/.local/share/bergamot/models/browsermt
PREFIX=$BROWSERMT/ende.student.tiny11

MODEL=model.intgemm.alphas.bin
VOCAB=vocab.deen.spm
SHORTLIST=lex.s2t.bin

set -x
./build/bin/slimt \
  --root ${PREFIX} \
  --model ${MODEL} \
  --vocab ${VOCAB} \
  --shortlist ${SHORTLIST} \
  < data/sample.txt

exit

export SLIMT_BLOB_PATH=$(realpath blobs)
export SLIMT_TRACE=1
SLIMT_EPS=1e-5 SLIMT_DEBUG=1 INTGEMM_CPUID=AVX512VNNI ./build/bin/slimt_test integration 2>&1
# SLIMT_EPS=1e-5 SLIMT_DEBUG=1 INTGEMM_CPUID=AVX512VNNI ./build/bin/slimt_test integration 2>&1
