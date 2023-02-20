#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly BIN_PATH=../../bin/
readonly GUESS=guess
readonly RING=ring
readonly BUILD=build

test -d "${BIN_PATH}" || mkdir "${BIN_PATH}"

if [ ! -f "${BIN_PATH}/${GUESS}" ]; then
  mkdir "${BUILD}"
  cd "${BUILD}"
  cmake cmake ../
  cmake --build . -j `nproc`
  cd -
  mv "${BUILD}/${GUESS}" "${BIN_PATH}"
  rm -rf "${BUILD}"
  echo "Guess build OK!"
fi

if [ ! -f "${BIN_PATH}/${RING}" ]; then
  g++ ring.cc -o "${BIN_PATH}/${RING}"
  echo "Ring build OK!"
fi