#!/usr/bin/env python3
"""Drive lds_probe under rocprofv3 and collect LDS counters per address pattern.

  probe.py run <op> <pattern file> <out.json> [--reps N] [--bin PATH] [--lock FILE]
  probe.py gen pairs <delta> [--size S]        all lane pairs (i<j): i at 0, j at delta
  probe.py gen delta <i> <j> <d0> <d1> <step>  lanes i, j only; j at delta d0..d1
  probe.py show <out.json>...                   print per-pattern counters
  probe.py sets <out.json>...                   conflict sets from a `gen pairs` run
  probe.py gen basic <size>                     baselines: none, one lane, broadcast, strides
  probe.py gen random <align> <n> <span> <seed> random aligned offsets in [0, span)
  probe.py gen raw                              copy patterns from stdin (validation)

Patterns go to stdout, one per line: 64 byte offsets (-1 = inactive lane) and
a '# label'. `run` launches lds_probe once (one dispatch per pattern) under
rocprofv3, holding the GPU measurement lock for that one process, and writes a
JSON list of {label, offsets, counters} with counters per dispatch, divided by
reps (so 1.0 means one cycle per instruction). Counters whose value does not
divide evenly are kept as floats; a non-integer per-instruction value means the
dispatch had overhead or noise and is worth looking at.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

COUNTERS = ["SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE", "SQ_LDS_ADDR_CONFLICT",
            "SQ_LDS_UNALIGNED_STALL", "SQ_INSTS_LDS"]
HERE = os.path.dirname(os.path.abspath(__file__))


def emit(offsets, label):
    print(" ".join(str(o) for o in offsets) + "  # " + label)


def gen(args):
    if args.kind == "pairs":
        d = int(args.a[0])
        for i in range(64):
            for j in range(i + 1, 64):
                p = [-1] * 64
                p[i], p[j] = 0, d
                emit(p, f"pair {i} {j} delta {d}")
    elif args.kind == "delta":
        i, j, d0, d1, step = map(int, args.a)
        for d in range(d0, d1 + 1, step):
            p = [-1] * 64
            p[i], p[j] = 0, d
            emit(p, f"pair {i} {j} delta {d}")
    elif args.kind == "basic":  # baselines for an access of <size> bytes per lane
        s = int(args.a[0])
        emit([-1] * 64, "none")
        emit([0] + [-1] * 63, "one lane")
        emit([0] * 64, "all lanes one address")
        for stride in sorted({s, 4, 8, 16, 12, 32, 64, 128, 256} - {0}):
            if stride >= s:
                emit([stride * l for l in range(64)], f"all lanes stride {stride}")
                emit([stride * l if l < 32 else -1 for l in range(64)], f"lanes 0-31 stride {stride}")
                emit([stride * l if l >= 32 else -1 for l in range(64)], f"lanes 32-63 stride {stride}")
                emit([stride * l if l < 16 else -1 for l in range(64)], f"lanes 0-15 stride {stride}")
                emit([stride * l if l < 8 else -1 for l in range(64)], f"lanes 0-7 stride {stride}")
    elif args.kind == "random":  # <align> <count> <span> <seed>: random aligned offsets
        import random
        align, count, span, seed = map(int, args.a)
        rng = random.Random(seed)
        for k in range(count):
            active = rng.choice([0.25, 0.5, 0.9, 1.0])
            sp = rng.choice([span // 8, span // 2, span])
            p = [rng.randrange(0, max(sp // align, 1)) * align if rng.random() < active else -1
                 for _ in range(64)]
            emit(p, f"random {seed}.{k}")
    elif args.kind == "raw":
        for line in sys.stdin:
            if line.strip() and not line.startswith("#"):
                print(line.rstrip())
    else:
        sys.exit(f"unknown generator {args.kind}")


def read_patterns(path):
    pats = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        body, _, label = line.partition("#")
        pats.append((list(map(int, body.split()[:64])), label.strip()))
    return pats


def run(args):
    pats = read_patterns(args.patterns)
    rocprof = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm-7.2.0"), "bin", "rocprofv3")
    with tempfile.TemporaryDirectory(prefix="lds_probe_") as tmp:
        # timeout: a hung dispatch must not hold the shared GPU lock
        cmd = ["flock", args.lock, "timeout", "-s", "KILL", str(args.timeout), rocprof, "--pmc", *COUNTERS, "-d", tmp, "-o", "r",
               "--output-format", "csv", "--", os.path.abspath(args.bin), args.op, str(args.reps),
               os.path.abspath(args.patterns)]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
            sys.exit(f"rocprofv3 failed ({r.returncode})")
        per = {}
        for row in csv.DictReader(open(os.path.join(tmp, "r_counter_collection.csv"))):
            if not row["Kernel_Name"].startswith("k_"):
                continue  # hipMemcpy's copy kernels
            per.setdefault(int(row["Dispatch_Id"]), {})[row["Counter_Name"]] = float(row["Counter_Value"])
    disp = [per[k] for k in sorted(per)]
    if len(disp) != len(pats):
        sys.exit(f"{len(disp)} probe dispatches for {len(pats)} patterns")
    out = []
    for (offs, label), c in zip(pats, disp):
        out.append({"label": label, "offsets": offs,
                    "counters": {k: (v / args.reps if v % args.reps else int(v) // args.reps)
                                 for k, v in c.items()}})
    json.dump({"op": args.op, "reps": args.reps, "results": out}, open(args.out, "w"))
    print(f"{args.op}: {len(out)} patterns -> {args.out}")


def show(args):
    for path in args.files:
        d = json.load(open(path))
        print(f"== {d['op']} (reps {d['reps']}; per instruction: BANK_CONFLICT IDX_ACTIVE ADDR_CONFLICT UNALIGNED INSTS)")
        for r in d["results"]:
            c = r["counters"]
            print(f"  {r['label']:<32} " + " ".join(f"{c.get(k, '-'):>6}" for k in COUNTERS))


def ranges(lanes):
    out, start = [], lanes[0]
    for a, b in zip(lanes, lanes[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}..{a}" if a > start else f"{a}")
            start = b
    return ",".join(out)


def sets(args):
    """Conflict sets from a `gen pairs` run: connected components of the
    relation 'lanes i and j conflict', and whether each is a clique."""
    for path in args.files:
        d = json.load(open(path))
        conf = {}
        for r in d["results"]:
            f = r["label"].split()
            conf[(int(f[1]), int(f[2]))] = r["counters"]["SQ_LDS_BANK_CONFLICT"]
        parent = list(range(64))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for (i, j), v in conf.items():
            if v:
                parent[find(i)] = find(j)
        comps = {}
        for lane in range(64):
            comps.setdefault(find(lane), []).append(lane)
        comps = sorted(comps.values())
        clique = all(conf.get((a, b), 0) for c in comps for x, a in enumerate(c) for b in c[x + 1:])
        vals = sorted({v for v in conf.values() if v})
        print(f"{d['op']}: {len(comps)} sets, sizes {sorted({len(c) for c in comps})}, "
              f"extra cycles per conflicting pair {vals or [0]}" + ("" if clique else ", NOT A PARTITION"))
        for c in comps:
            print("    {" + ranges(c) + "}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("kind")
    g.add_argument("a", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("op")
    r.add_argument("patterns")
    r.add_argument("out")
    r.add_argument("--reps", type=int, default=64)
    r.add_argument("--bin", default=os.path.join(HERE, "lds_probe"))
    r.add_argument("--lock", default="/tmp/gpu-measure.lock")
    r.add_argument("--timeout", type=int, default=120, help="seconds, per lds_probe process")
    s = sub.add_parser("show")
    s.add_argument("files", nargs="+")
    t = sub.add_parser("sets")
    t.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"gen": gen, "run": run, "show": show, "sets": sets}[args.cmd](args)


if __name__ == "__main__":
    main()
