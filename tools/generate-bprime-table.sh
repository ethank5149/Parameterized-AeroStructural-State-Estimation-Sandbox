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
#
# MATERIAL=pica overrides the generator's pyrolysis-gas element composition
# with PICA's. That composition is FIAT's own: Rabinovitch, Marx & Blanquart
# (AIAA 2014-2246) §III.A.2 report Milos & Chen's FIATv3 PICA deck injecting
# C:H:N:O:Si = 0.18:0.68:0.014:0.12:0.006 by mole. The trace N and Si are
# dropped because the species set below carries no silicon, and the value is
# renormalised over C/H/O. Keep it in step with
# passes.thermal.fiat.pica_kinetics.PICA_PYROLYSIS_ELEMENTS.
set -euo pipefail

MPP_SRC=${MPP_SRC:-/config/Mutationpp-1.0.5}
BPRIME_SRC=${BPRIME_SRC:-/config/Modified_Bprime-main}
BUILD=${BUILD:-$(mktemp -d)}
MATERIAL=${MATERIAL:-tacot26}

case "$MATERIAL" in
  tacot26) PYRO="N:0.000, O: 0.115, C: 0.206, H: 0.679" ;;
  pica)    PYRO="N:0.000, O: 0.122449, C: 0.183673, H: 0.693878" ;;
  *) echo "MATERIAL must be tacot26 or pica, got '$MATERIAL'" >&2; exit 2 ;;
esac

OUT=${OUT:-data/bprime/${MATERIAL}-air.dat}

CXX=${CXX:?set CXX to a C++11 compiler}

cmake -S "$MPP_SRC" -B "$BUILD/mpp" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BUILD/install" -DBUILD_TESTING=OFF
cmake --build "$BUILD/mpp" --target install

# The generator's shipped CMakeLists hardcodes an install lib64 path, so it
# is compiled directly against the tree we just built.
mkdir -p "$BUILD/gen" && cp "$BPRIME_SRC"/*.cpp "$BPRIME_SRC"/tacot26.xml "$BUILD/gen/"

# Rewrite only the pyrolysis-gas composition; the edge composition (air) and
# the species set are shared between the two materials.
python3 - "$BUILD/gen/tacot26.xml" "$PYRO" <<'PY'
import re, sys
path, pyro = sys.argv[1], sys.argv[2]
text = open(path).read()
new, n = re.subn(
    r'(<composition name="pyro">)[^<]*(</composition>)',
    lambda m: m.group(1) + " " + pyro + " " + m.group(2),
    text,
)
if n != 1:
    raise SystemExit(f"expected exactly one pyro composition, found {n}")
open(path, "w").write(new)
PY
grep -n 'name="pyro"' "$BUILD/gen/tacot26.xml"
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
