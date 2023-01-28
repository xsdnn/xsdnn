#!/bin/bash

git submodule init
git submodule update

mkdir third_party
mv json/ third_party/json
mv logger/ third_party/spdlog

DIRECTORY=./eigen/Eigen
SOURCE_DIR=$( pwd; )

if [ ! -d "$DIRECTORY" ]; then
  echo "$DIRECTORY does not exist."
  git clone https://gitlab.com/libeigen/eigen.git
  echo "export EIGEN3_INCLUDE_DIR='$SOURCE_DIR/eigen/'" >> ~/.bashrc
else
  echo "Eigen finded"
fi