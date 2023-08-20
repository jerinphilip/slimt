#!/bin/bash

TAGS=(
  ende
  enfr
)

set -x
export INTGEMM_CPUID=AVX512VNNI
BERGAMOT="/home/jerin/code/bergamot-translator/build"
BROWSERMT="/home/jerin/.local/share/bergamot/models/browsermt"
mkdir -p traces

for tag in ${TAGS[@]}; do
  export DEBUG_VARIABLES_SAVE_PATH="/home/jerin/code/slimt/blobs/$tag"
  rm -r $DEBUG_VARIABLES_SAVE_PATH
  mkdir -p $DEBUG_VARIABLES_SAVE_PATH
  $BERGAMOT/app/bergamot --model-config-paths $BROWSERMT/$tag.student.tiny11/config.bergamot.yml --log-level off \
    < data/numbers2x3.txt \
    &> traces/$tag.txt
done
