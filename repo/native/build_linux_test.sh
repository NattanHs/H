#!/bin/sh
# Compila a versão SEM JUCE para testar o núcleo no Linux.   uso: ./build_linux_test.sh [python]
set -e
PY=${1:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/build_linux}
mkdir -p "$OUT"
EXT=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
PB=${PYBIND11_DIR:-$HERE/../third_party/pybind11}
g++ -std=c++17 -O2 -fPIC -shared -Wall -Wextra -I"$HERE/src" -I"$INC" -I"$PB/include" \
  "$HERE/src/core.cpp" "$HERE/src/bindings.cpp" "$HERE/src/device_stub.cpp" -o "$OUT/_native$EXT"
echo "gerado: $OUT/_native$EXT"
