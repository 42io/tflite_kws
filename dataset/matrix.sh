#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly DATASET_FILE_NAME=$1
readonly DATASET_NUM_OUTPUT=`awk '{if(x<$1)x=$1}END{print ++x/3}' "${DATASET_FILE_NAME}"`

bash ./../src/brain/build.sh

do_confusion_matrix() {
  local -r labels=("zero" "one" "two" "three" "four" "five" "six" "seven" "eight" "nine" "#unk#" "#pub#")
  local model=$1
  local start=$2
  local count=$3
  local delay=$4
  local glean=$5
  local i
  for i in `seq ${DATASET_NUM_OUTPUT}` ; do
    echo -ne "${labels[i-1]}\t"
    awk -v m="${DATASET_NUM_OUTPUT}" '$1 >= m' "${DATASET_FILE_NAME}" \
      | awk -v i="${i}" -v m="${DATASET_NUM_OUTPUT}" '$1 == i - 1 + m || $1 == i - 1 + 2*m' \
      | awk -v s="${start}" -v c="${count}" -v d="${delay}" '{for(i=0;i<c;i++)print $(i+s);for(i=0;i<d;i++)print 0}' \
      | ./../bin/guess "./../models/${model}" \
      | awk -v g="${glean}" 'NR % g == 0' \
      | awk '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;} for(i=1;i<=NF;i++){if(i>1)printf " ";printf "%d", i==j} print ""}' \
      | awk '{for(i=1;i<=NF;i++)sum[i]+=$i} END {for(j=1;j<i;j++){if(j>1)printf " ";printf "%.2f", sum[j]/NR} print " | " NR}'
  done
}

do_validation() {
  local model=$1
  local start=$2
  local count=$3
  local delay=$4
  local glean=$5
  local i
  for i in `seq ${DATASET_NUM_OUTPUT}` ; do
    awk -v m="${DATASET_NUM_OUTPUT}" '$1 >= m' "${DATASET_FILE_NAME}" \
      | awk -v i="${i}" -v m="${DATASET_NUM_OUTPUT}" '$1 == i - 1 + m || $1 == i - 1 + 2*m' \
      | awk -v s="${start}" -v c="${count}" -v d="${delay}" '{for(i=0;i<c;i++)print $(i+s);for(i=0;i<d;i++)print 0}' \
      | ./../bin/guess "./../models/${model}" \
      | awk -v g="${glean}" 'NR % g == 0' \
      | awk -v x="${i}" '{m=$1;j=1;for(i=j;i<=NF;i++)if($i>m){m=$i;j=i;} if(j!=x)print x}'
  done
}

do_all() {
  local title=$1
  local model=$2
  local start=$3
  local count=$4
  local delay=$5
  local glean=$6
  echo "${title} confusion matrix..."
  do_confusion_matrix ${model} ${start} ${count} ${delay} ${glean} | sed 's!0.00 ! .   !g'
  echo "${title} guessed wrong `do_validation ${model} ${start} ${count} ${delay} ${glean} | wc -l`..."
}

do_all 'DCNN47'  'dcnn47.tflite'  15 611 0  1
do_all 'DCNN13'  'dcnn13.tflite'  15 611 65 52
do_all 'ECNN47'  'ecnn47.tflite'  15 611 0  1
do_all '2ECNN47' '2ecnn47.tflite' 15 611 0  1
do_all '2ECNN13' '2ecnn13.tflite' 15 611 65 52

# test
# do_all '_2ECNN47' '_2ecnn47.tflite' 15 611 0  1