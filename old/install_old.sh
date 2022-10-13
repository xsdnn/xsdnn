#!/bin/bash
DIRECTORY=./eigen/Eigen
SOURCE_DIR=$( pwd; )

if [ ! -d "$DIRECTORY" ]; then
  echo "$DIRECTORY does not exist."
  git clone https://gitlab.com/libeigen/eigen.git 
  echo "export EIGEN3_INCLUDE_DIR='$SOURCE_DIR/eigen/'" >> ~/.bashrc 
else 
  echo "Eigen finded"
fi


