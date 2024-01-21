#!/bin/bash

export DEBUG_VARIABLES_SAVE_PATH="/home/jerin/code/slimt/blobs/ml-xlit"
mkdir -p $DEBUG_VARIABLES_SAVE_PATH
rm $DEBUG_VARIABLES_SAVE_PATH/*

/home/jerin/code/bergamot-translator/build/app/bergamot \
  --model-config-paths $HOME/code/slimt-t12n/outputs/mal-eng/model.nano.npz.decoder.yml \
  --log-level off \
  < data/ml-xlit.txt \
  2> traces/ml-xlit.trace.txt
