#!/usr/bin/env python3
"""Candidate LDS bank-conflict models, checked against lds_probe measurements.

A model says, for an op: which lanes are serviced together (groups, one
baseline cycle each), how many banks the address is spread over (bank =
dword index mod banks), and the bytes and addresses per lane. Predicted extra
cycles for a pattern = sum over groups of (max over banks of the number of
distinct dwords that group's active lanes touch in that bank) - 1.
read2/write2 are two such accesses back to back.

  model.py check <arch> <result.json>...   compare predictions with measured
                                           SQ_LDS_BANK_CONFLICT per pattern
Patterns that hit SQ_LDS_UNALIGNED_STALL are skipped (misaligned accesses take
another path that the bank-conflict counter does not see).
"""
import json
import sys


def blocks(*starts, size):
    return [[l for s in g for l in range(s, s + size)] for g in starts]


def halves():
    return [list(range(0, 32)), list(range(32, 64))]


def consecutive(n):
    return [list(range(a, a + n)) for a in range(0, 64, n)]


# Lane groups measured with `probe.py gen pairs 512` (2016 pairs per op).
CDNA_READ_B128_8 = blocks((0, 20), (4, 16), (8, 28), (12, 24), (32, 52), (36, 48), (40, 60), (44, 56), size=4)
GFX950_READ_B128_16 = blocks((0, 12, 20, 24), (4, 8, 16, 28), (32, 44, 52, 56), (36, 40, 48, 60), size=4)

# op class -> (groups, banks)
GFX950 = {
    "narrow": (halves(), 32),          # <=4-byte reads and writes
    "read_b64": (halves(), 64),
    "read_b96": (CDNA_READ_B128_8, 32),
    "read_b128": (GFX950_READ_B128_16, 64),
    "write_b64": (consecutive(16), 32),
    "write_b96": (consecutive(8), 32),
    "write_b128": (consecutive(8), 32),
    "atomic_b32": (halves(), 32),
    "atomic_b64": (consecutive(16), 32),
    # compare-swap carries two data dwords per lane: 16 lanes a cycle and a
    # 64-byte period, i.e. behaves as 16 banks
    "atomic_cmpst_b32": (consecutive(16), 16),
    # float atomics run lane by lane (3 cycles per active lane in
    # SQ_LDS_IDX_ACTIVE); SQ_LDS_BANK_CONFLICT stays 0 for any addresses
    "atomic_float": ([], 32),
    "read_b64_tr": (halves(), 64),
    "read_b96_tr": (consecutive(16), 64),
}
GFX90A = {  # from the 2026-11-08 model in omniprobe; to be re-measured with this tool
    "narrow": (halves(), 32),
    "read_b64": (consecutive(16), 32),
    "read_b96": (None, 32),
    "read_b128": (CDNA_READ_B128_8, 32),
    "write_b64": (consecutive(16), 32),
    "write_b96": (None, 32),
    "write_b128": (consecutive(8), 32),
}
MODELS = {"gfx950": GFX950, "gfx90a": GFX90A}


def op_shape(op):
    """(class, bytes per address, list of extra byte offsets per address)."""
    if op.startswith(("add_", "max_", "pk_add", "wrxchg", "cmpst")):  # RMW
        if op.startswith(("add_f32", "pk_add")):
            return "atomic_float", 4, [0]
        if op.startswith("cmpst"):
            return ("atomic_cmpst_b64" if "64" in op else "atomic_cmpst_b32"), (8 if "64" in op else 4), [0]
        if op.startswith("wrxchg2"):
            return "atomic_b32", 4, [0, 4 * int(op.rsplit("_", 1)[1])]
        return ("atomic_b64" if "64" in op else "atomic_b32"), (8 if "64" in op else 4), [0]
    if "_tr_" in op:
        return ("read_b96_tr" if "b96" in op else "read_b64_tr"), (12 if "b96" in op else 8), [0]
    rw = "read" if op.startswith("read") else "write"
    if op.startswith(("read2", "write2")):
        size = 4 if "_b32" in op else 8
        o1 = int(op.rsplit("_", 1)[1]) * size * (64 if "st64" in op else 1)
        cls = "narrow" if size == 4 else f"{rw}_b64"
        if size == 8 and rw == "read":
            cls = "read2_b64"
        return cls, size, [0, o1]
    tag = op.split("_")[1]  # read_u8_d16_hi -> u8
    size = {"u8": 1, "i8": 1, "b8": 1, "u16": 2, "i16": 2, "b16": 2, "b32": 4,
            "b64": 8, "b96": 12, "b128": 16}[tag]
    return ("narrow" if size <= 4 else f"{rw}_{tag}"), size, [0]
    raise ValueError(op)


def extra(groups, banks, offsets, size, atomic=False):
    """Atomics serialize lanes even on one address (no broadcast): count
    (lane, dword) pairs per bank instead of distinct dwords."""
    total = 0
    for g in groups:
        per_bank = {}
        for lane in g:
            if offsets[lane] < 0:
                continue
            first, last = offsets[lane] // 4, (offsets[lane] + size - 1) // 4
            for dw in range(first, last + 1):
                per_bank.setdefault(dw % banks, set()).add((lane, dw) if atomic else dw)
        if per_bank:
            total += max(len(s) for s in per_bank.values()) - 1
    return total


def predict(model, op, offsets):
    cls, size, extras = op_shape(op)
    if cls == "read2_b64":  # each half of a read2_b64 behaves like a b64 write
        cls = "write_b64"
    groups, banks = model[cls]
    if groups is None:
        return None
    return sum(extra(groups, banks, [o + e if o >= 0 else -1 for o in offsets], size, cls.startswith("atomic"))
               for e in extras)


def check(arch, paths):
    model = MODELS[arch]
    bad_total = 0
    for path in paths:
        d = json.load(open(path))
        n = bad = skipped = 0
        first_bad = []
        for r in d["results"]:
            c = r["counters"]
            if c["SQ_LDS_UNALIGNED_STALL"]:
                skipped += 1
                continue
            p = predict(model, d["op"], r["offsets"])
            n += 1
            if p != c["SQ_LDS_BANK_CONFLICT"]:
                bad += 1
                if len(first_bad) < 3:
                    first_bad.append(f"{r['label']}: predicted {p}, measured {c['SQ_LDS_BANK_CONFLICT']}")
        bad_total += bad
        print(f"{path}: {d['op']}: {n - bad}/{n} match" + (f" ({skipped} unaligned skipped)" if skipped else ""))
        for s in first_bad:
            print("    " + s)
    return 1 if bad_total else 0


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "check":
        sys.exit(__doc__)
    sys.exit(check(sys.argv[2], sys.argv[3:]))
