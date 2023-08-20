#!/bin/bash
# For use in debugging.

export SLIMT_BLOB_PATH=$(realpath blobs/enfr)
export SLIMT_TRACE=1

./build/bin/slimt \
  --root /home/jerin/.local/share/bergamot/models/browsermt/enfr.student.tiny11 \
  --model model.intgemm.alphas.bin --vocabulary vocab.fren.spm --shortlist lex.s2t.bin \
  < data/numbers2x3.txt
