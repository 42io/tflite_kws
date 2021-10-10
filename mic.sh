#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly MODEL=$1

test -f bin/guess || bash ./src/brain/build.sh
test -f dataset/dataset/google_speech_commands/bin/fe || \
        bash ./dataset/dataset/google_speech_commands/src/features/build.sh

argmax() { mawk -Winteractive '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;}print j-1}'; }
stable() { mawk -Winteractive -v u=$1 '{if(x!=$1){c=0;x=$1}else if(++c==u&&y!=x)print y=x}'; }
ignore() { mawk -Winteractive -v t=$1 '{if($1<t)print $1}'; }

fe() { dataset/dataset/google_speech_commands/bin/fe; }

if [[ '47.tflite' == ${MODEL: -9} ]]
then
  arecord -f S16_LE -c1 -r16000 -t raw | fe | \
    bin/ring 47 | bin/guess ${MODEL} | argmax | stable 10 | ignore 10
elif [[ '13.tflite' == ${MODEL: -9} ]]
then
  arecord -f S16_LE -c1 -r16000 -t raw | fe | \
    bin/guess ${MODEL} | argmax | stable 10 | ignore 10
else
  echo "bad model"
fi
