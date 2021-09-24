#!/usr/bin/env bash

SRC_DIR="$1"
BUILD_DIR="$2"

mkdir -p "${BUILD_DIR}/tests/midas"

/usr/bin/protoc --cpp_out "${BUILD_DIR}/tests/midas" \
  -I "${SRC_DIR}/tests/midas" \
  -I "${SRC_DIR}/include" \
  "${SRC_DIR}/include/midas/ProtobufSupport.proto" \
  "${SRC_DIR}/tests/midas/MidasTests.proto"
