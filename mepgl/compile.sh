#!/bin/bash

export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

DEBUG=

while [ "$1" != "" ]; do
    case $1 in
        -d | --debug )           shift
                                DEBUG=1
                                ;;
    esac
    shift
done


if [ "$DEBUG" = "1" ]; then
	echo "==> Builidin debug binaries"
    echo ""

    mkdir -p ./build/debug
    cd ./build/debug
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Debug ../../src
    make
    cd ../../
    cp -f ./build/debug/mepgl ./mepgl

else
	echo "==> Builiding release binaries"
    echo ""

    mkdir -p ./build/release
    cd ./build/release
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -DCMAKE_BUILD_TYPE=Release ../../src
    make
    cd ../../
    cp -f ./build/release/mepgl ./mepgl
fi
