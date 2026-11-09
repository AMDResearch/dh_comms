# lds_bank_probe: measure how the LDS services each DS instruction

Tools to measure, on the GPU present, which lanes of a wave the LDS services
together and how addresses map to banks, for every LDS access class, and to
check a bank-conflict model against those measurements. The model feeds the
LDS bank-conflict analysis in Omniprobe's memory-analysis handler.

The method is Rene van Oostrum's pairwise-lane method, generalized: run one
DS instruction with chosen per-lane addresses (other lanes inactive), one
dispatch per address pattern, and read hardware counters per dispatch.

| file | what |
|---|---|
| `lds_probe.cpp` | HIP program. `lds_probe <op> <reps> <pattern file>`: one dispatch per pattern; each active lane runs `<op>` (inline asm, so the instruction is exact) `<reps>` times. `--list` prints the ops this GPU supports. |
| `probe.py` | `gen` writes patterns (`basic`, `pairs`, `delta`, `random`, `raw`); `run` runs lds_probe under rocprofv3 and the GPU lock and stores counters per pattern, per instruction; `show` and `sets` print them. |
| `model.py` | Bank-conflict models per architecture; `check <arch> <results>` compares predicted extra cycles with the measured `SQ_LDS_BANK_CONFLICT` for every pattern. |
| `run_all.sh` | The whole campaign for all ops: build, measure, derive conflict sets, check the model. About 5 minutes on gfx950. |

```
tools/lds_bank_probe/run_all.sh /tmp/lds-meas            # all ops
tools/lds_bank_probe/run_all.sh /tmp/lds-meas read_b64   # some ops
```

Counters collected per dispatch: `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_IDX_ACTIVE`,
`SQ_LDS_ADDR_CONFLICT`, `SQ_LDS_UNALIGNED_STALL`, `SQ_INSTS_LDS`. Values are
divided by reps, so they are per instruction. `SQ_LDS_IDX_ACTIVE` (cycles the
LDS is busy) is always the baseline cycles of the op plus
`SQ_LDS_BANK_CONFLICT` plus `SQ_LDS_UNALIGNED_STALL`. That makes the baseline
(how many lane groups the LDS processes) directly measurable.

Ops: reads `u8 i8 u16 *_d16 *_d16_hi b32 b64 b96 b128`, writes `b8 b16
*_d16_hi b32 b64 b96 b128`, `read2`/`write2`/`st64` in `b32` and `b64` with
several offsets, atomics `add_u32 add_u64` with and without return, and the
gfx950 transpose loads `read_b64_tr_b{4,8,16}` and `read_b96_tr_b6`.

## Notes

- Use naturally aligned offsets (`b96` needs 16 bytes). Misaligned accesses
  take an unaligned path: `SQ_LDS_UNALIGNED_STALL` cycles, not bank conflicts.
  `model.py` skips those patterns.
- A misaligned `ds_add_u64` (4- but not 8-byte aligned) never completes on
  gfx950, so `run_all.sh` keeps atomic patterns aligned, and `probe.py run`
  kills lds_probe after `--timeout` seconds so a hang cannot hold the GPU lock.
- Build into and run from local disk (`/tmp`): code objects on some network
  filesystems cannot be mmapped.
