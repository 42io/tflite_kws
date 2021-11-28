#!/bin/bash

set -e
set -u

cd "`dirname "${BASH_SOURCE[0]}"`"

export LC_ALL=C

readonly BIN_PATH=../../bin/
readonly FE=fe

test -d "${BIN_PATH}" || mkdir "${BIN_PATH}"

if [ ! -f "${BIN_PATH}/${FE}" ]; then
  ./../../dataset/dataset/google_speech_commands/src/features/build.sh
  ln -s ./../dataset/dataset/google_speech_commands/bin/fe "${BIN_PATH}/${FE}"
fi