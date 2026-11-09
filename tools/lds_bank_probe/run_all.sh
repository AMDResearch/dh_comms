#!/bin/bash
# Measure how the LDS of the GPU present services every DS access class, and
# check the result against model.py. Reproduces the whole campaign:
#   tools/lds_bank_probe/run_all.sh <output dir> [ops...]
# For each op (default: all ops lds_probe --list offers on this GPU):
#   basic   baselines: one lane, broadcast, full/half/quarter wave at strides
#   pairs   all 2016 lane pairs, the two lanes 512 bytes apart (same bank for
#           any bank count up to 128): which lanes are serviced together
#   delta   lanes 0 and 1, the second at 0..1024 bytes: bank function
#   random  300 patterns of random aligned offsets: validation of the model
# Each lds_probe process runs under flock on the GPU measurement lock
# (LDS_PROBE_LOCK, default /tmp/gpu-measure.lock) with a kill timeout.
# The binary is built into the output dir (some filesystems cannot mmap code
# objects, so keep the output dir on local disk, e.g. /tmp).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: run_all.sh <output dir> [ops...]}"; shift
ROCM="${ROCM_PATH:-/opt/rocm-7.2.0}"
LOCK="${LDS_PROBE_LOCK:-/tmp/gpu-measure.lock}"
mkdir -p "$OUT/res" "$OUT/pat" || exit 1
ARCH="$("$ROCM/bin/rocminfo" | sed -n 's/^ *Name: *\(gfx[0-9a-z]*\) *$/\1/p' | head -1)"
echo "GPU: $ARCH"
"$ROCM/bin/hipcc" -O2 --offload-arch="$ARCH" -o "$OUT/lds_probe" "$HERE/lds_probe.cpp" || exit 1
BIN="$OUT/lds_probe"
P="python3 -I $HERE/probe.py"

align() { # natural alignment of an op's access; b96 needs 16
    case $1 in
        *_tr_b*) case $1 in *b96*) echo 16;; *) echo 8;; esac;;
        add_*|max_*|pk_add*|wrxchg*|cmpst*) case $1 in *64*) echo 8;; *) echo 4;; esac;;
        *u8*|*i8*|*b8*) echo 1;; *u16*|*i16*|*b16*) echo 2;;
        *b32*|*u32) echo 4;; *b64*|*u64) echo 8;; *b96*|*b128*) echo 16;;
    esac
}
size() { case $1 in *b96*) echo 12;; *) align "$1";; esac; }

$P gen pairs 512 > "$OUT/pat/pairs512.pat"
[ $# -gt 0 ] && OPS="$*" || OPS="$("$BIN" --list)"
fail=0
for op in $OPS; do
    a=$(align "$op"); s=$(size "$op")
    # Misaligned atomics hang on gfx950: keep atomic and transpose patterns aligned.
    $P gen basic "$s" | { case $op in add_*|max_*|pk_add*|wrxchg*|cmpst*|*_tr_*) grep -v "stride 12";; *) cat;; esac; } > "$OUT/pat/basic_$op.pat"
    step=$a; [ "$step" -lt 4 ] && { $P gen delta 0 1 0 300 "$step" > "$OUT/pat/delta_$op.pat"; true; } \
        || $P gen delta 0 1 0 1024 "$step" > "$OUT/pat/delta_$op.pat"
    $P gen random "$a" 300 4096 7 > "$OUT/pat/random_$op.pat"
    for kind in basic pairs delta random; do
        pat="$OUT/pat/${kind}_$op.pat"; [ $kind = pairs ] && pat="$OUT/pat/pairs512.pat"
        $P run "$op" "$pat" "$OUT/res/${kind}_$op.json" --bin "$BIN" --lock "$LOCK" >/dev/null \
            || { echo "FAILED: $kind $op"; fail=1; }
    done
    $P sets "$OUT/res/pairs_$op.json"
done
echo "== model check ($ARCH)"
python3 -I "$HERE/model.py" check "$ARCH" "$OUT"/res/*.json > "$OUT/model-check.txt"
rc=$?
python3 -I -c 'import re,sys
for l in open(sys.argv[1]):
    m = re.search(r"(\d+)/(\d+) match", l)
    if not m or m.group(1) != m.group(2): print(l.rstrip())' "$OUT/model-check.txt"
echo "model check: $([ $rc -eq 0 ] && echo all patterns match || echo MISMATCHES, see $OUT/model-check.txt)"
[ $fail -eq 0 ] && exit $rc || exit 1
