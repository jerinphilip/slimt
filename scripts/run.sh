#!/bin/bash

BROWSERMT=$HOME/.local/share/bergamot/models/browsermt
PREFIX=$BROWSERMT/ende.student.tiny11
FOLLOW_PREFIX=$BROWSERMT/deen.student.tiny11

MODEL=model.intgemm.alphas.bin
VOCAB=vocab.deen.spm
SHORTLIST=lex.s2t.bin

ARGS=(
  --root ${PREFIX}
  --model ${MODEL}
  --vocabulary ${VOCAB}
  --shortlist ${SHORTLIST}

  --follow-root ${FOLLOW_PREFIX}
  --follow-model ${MODEL}
  --follow-vocabulary ${VOCAB}
  --follow-shortlist ${SHORTLIST}

  --async
  --workers 24
  --poll 1
)

set -x
./build/app/slimt-cli "${ARGS[@]}" < data/wngt20/sources.shuf.10k > /dev/null

exit

export SLIMT_BLOB_PATH=$(realpath blobs)
export SLIMT_TRACE=1
SLIMT_EPS=1e-5 SLIMT_DEBUG=1 INTGEMM_CPUID=AVX512VNNI ./build/bin/slimt_test integration 2>&1
# SLIMT_EPS=1e-5 SLIMT_DEBUG=1 INTGEMM_CPUID=AVX512VNNI ./build/bin/slimt_test integration 2>&1
