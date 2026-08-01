#!/usr/bin/env bash
# Regenerate the TACOT-26 equilibrium B' table used by passes.thermal.fiat.
#
# Requires Mutation++ (open source) and the Modified_Bprime generator of
# Padovan et al., Int. J. Heat and Mass Transfer, 2023. Neither is vendored
# here; set the two paths below to local checkouts.
#
# TACOT is the Theoretical Ablative Composite for Open Testing, the open
# surrogate for PICA. It is not PICA, and tables generated from it must not
# be reported as PICA surface chemistry.
set -euo pipefail

MPP_SRC=${MPP_SRC:-/config/Mutationpp-1.0.5}
BPRIME_SRC=${BPRIME_SRC:-/config/Modified_Bprime-main}
BUILD=${BUILD:-$(mktemp -d)}
OUT=${OUT:-data/bprime/tacot26-air.dat}

CXX=${CXX:?set CXX to a C++11 compiler}

cmake -S "$MPP_SRC" -B "$BUILD/mpp" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BUILD/install" -DBUILD_TESTING=OFF
cmake --build "$BUILD/mpp" --target install

# The generator's shipped CMakeLists hardcodes an install lib64 path, so it
# is compiled directly against the tree we just built.
mkdir -p "$BUILD/gen" && cp "$BPRIME_SRC"/*.cpp "$BPRIME_SRC"/tacot26.xml "$BUILD/gen/"
"$CXX" -O2 -std=c++11 "$BUILD/gen/generate_bprime_table.cpp" \
  -I"$BUILD/install/include/mutation++" -I"$BUILD/install/include" \
  -I"$MPP_SRC/thirdparty/eigen" \
  -L"$BUILD/install/lib" -lmutation++ -Wl,-rpath,"$BUILD/install/lib" \
  -o "$BUILD/gen/generate_bprime_table"

cd "$BUILD/gen"
MPP_DATA_DIRECTORY="$MPP_SRC/data" ./generate_bprime_table \
  -m tacot26.xml -bl edge -py pyro \
  -T 250:50:4000 -P 101.325:10:101325 -b 0.0:0.1:5.0 > table.dat
cd - >/dev/null
mkdir -p "$(dirname "$OUT")" && cp "$BUILD/gen/table.dat" "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") rows)"
