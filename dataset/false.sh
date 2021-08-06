#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly DATASET_DIR=$1
readonly DATASET_TAKE=$2

bash ./../src/brain/build.sh
bash ./../dataset/dataset/google_speech_commands/src/features/build.sh

argmax() { mawk -Winteractive '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;}print j-1}'; }
ignore() { mawk -Winteractive -v t=$1 '{if($1<t)print $1}'; }

find_guess() {
  local model=$1
  local ring=$2
  local skip=$3
  find "$DATASET_DIR" -name '*.wav' -print0 | sort -z | \
    head -z -n "$DATASET_TAKE" | xargs -0 -I{} bash -c \
      "dataset/google_speech_commands/bin/fe '{}' | \
       ../bin/ring $ring | \
       ../bin/guess ../models/$model.tflite | \
       tail +$skip"
}

false_positive() {
  local model=$1
  echo -e "${model^^}\t`find_guess "$@" | argmax | ignore 10 | wc -l` | `find_guess "$@" | wc -l`"
}

false_positive edcnn47  47  1
false_positive ecnn47   47  1
false_positive dcnn13   1   47
false_positive dcnn47   47  1
false_positive mlp      49  1 # doubtfully
false_positive cnn      49  1 # doubtfully
false_positive rnn      49  1 # doubtfully