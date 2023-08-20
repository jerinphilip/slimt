#!/bin/bash
# For use in debugging.

set -x

export SLIMT_TRACE=1
export SLIMT_EPS=1e-4
export SLIMT_DEBUG=1

INPUT_SAMPLE=data/numbers2x3.txt

function enfr() {
  export SLIMT_BLOB_PATH=$(realpath blobs/enfr)

  ./build/bin/slimt \
    --root /home/jerin/.local/share/bergamot/models/browsermt/enfr.student.tiny11 \
    --model model.intgemm.alphas.bin --vocabulary vocab.fren.spm --shortlist lex.s2t.bin \
    < $INPUT_SAMPLE
}

function ende() {
  export SLIMT_BLOB_PATH=$(realpath blobs/ende)

  ./build/bin/slimt \
    --root /home/jerin/.local/share/bergamot/models/browsermt/ende.student.tiny11 \
    --model model.intgemm.alphas.bin --vocabulary vocab.deen.spm --shortlist lex.s2t.bin \
    < $INPUT_SAMPLE
}

enfr
ende
