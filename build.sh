#!/usr/bin/env bash

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OS=$(uname -s)

if [ "$OS" == "Linux" ]; then
    DIR_OS="Linux"
else
    echo "Unsupported system!"
    exit 1
fi

python3 $DIR/tools/ci_build/build.py --build_dir $DIR/build/$DIR_OS "$@"
