#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly MODEL=$1

bash ./src/brain/build.sh
bash ./src/features/build.sh

argmax() { mawk -Winteractive '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;}print j-1}'; }
stable() { mawk -Winteractive -v u=$1 '{if(x!=$1){c=0;x=$1}else if(++c==u&&y!=x)print y=x}'; }
ignore() { mawk -Winteractive -v t=$1 '{if($1<t)print $1}'; }

if [[ '47.tflite' == ${MODEL: -9} ]]
then
  arecord -f S16_LE -c1 -r16000 -t raw | bin/fe | \
    bin/ring 47 | bin/guess ${MODEL} | argmax | stable 10 | ignore 10
elif [[ '13.tflite' == ${MODEL: -9} ]]
then
  arecord -f S16_LE -c1 -r16000 -t raw | bin/fe | \
    bin/guess ${MODEL} | argmax | stable 10 | ignore 10
else
  echo "bad model"
fi
