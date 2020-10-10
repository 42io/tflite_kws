#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly TFLIB=../../lib/build/libtensorflow-lite.a

if [ ! -f ${TFLIB} ]; then
  cd ../../lib/tensorflow/
  ./tensorflow/lite/tools/make/download_dependencies.sh
  ./tensorflow/lite/tools/make/build_lib.sh
  mkdir ../build/
  mv tensorflow/lite/tools/make/gen/*/lib/libtensorflow-lite.a ../build/
  rm -r tensorflow/lite/tools/make/gen/
  cd -
  echo "TFLite build OK!"
fi

mkdir -p ../../bin

g++ -O3 -fPIC --std=c++11 -Wno-error=unused-parameter -DTFLITE_WITHOUT_XNNPACK -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK -pthread \
  -I../../lib/tensorflow/ \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/ \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/eigen \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/absl \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/ruy \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/neon_2_sse \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
  -I../../lib/tensorflow/tensorflow/lite/tools/make/downloads/fp16/include \
  -o ../../bin/guess guess.cc ${TFLIB} \
  -lstdc++ -lpthread -lm -lz -ldl

echo "Guess build OK!"

g++ ring.cc -o ../../bin/ring

echo "Ring build OK!"