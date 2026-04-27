# Device API

## Responsibility

Device-side functions for submitting messages from GPU kernels to shared buffers accessible by host code. Covers vector submits (all active lanes), scalar submits (first lane only), and the internal sub-buffer management and synchronization primitives.

## Key Source Files

| File | Role |
|------|------|
| `include/dh_comms_dev.h` | All device-side submit functions and internal helpers |
| `include/data_headers_dev.h` | Device-side wire format structs (wave_header, lane_header) |
| `include/gpu_arch_constants.h` | GPU architecture constants (wave size, etc.) |

## Key Types and Classes

| Type | File | Role |
|------|------|------|
| `wave_header_t` | `data_headers_dev.h` | Per-wave metadata: exec mask, data size, timestamp, DWARF file/line/column, user type/data |
| `lane_header_t` | `data_headers_dev.h` | Per-lane metadata: thread coordinates |

## Key Functions and Entry Points

| Function | Location | Role |
|----------|----------|------|
| `v_submit_message(rsrc, msg, size, ...)` | `dh_comms_dev.h:255` | Vector submit: all active lanes write data |
| `s_submit_message(rsrc, msg, size, ...)` | `dh_comms_dev.h:279` | Scalar submit: first active lane writes data |
| `v_submit_address(rsrc, addr, ...)` | `dh_comms_dev.h:318` | Specialized address message (packs rw_kind, memory_space, sizeof_pointee into user_data) |
| `s_submit_wave_header(rsrc, ...)` | `dh_comms_dev.h:304` | Header-only submit, no data payload |
| `s_submit_time_interval(rsrc, ...)` | `dh_comms_dev.h:333` | Time interval message submit |
| `get_sub_buffer_idx(no_sub_buffers)` | `dh_comms_dev.h:36` | Compute sub-buffer for current wave (workgroup index % n) |
| `wave_acquire(flags, idx, lane)` | `dh_comms_dev.h:65` | Acquire exclusive sub-buffer access via atomic flag |
| `wave_release(flags, idx, lane)` | `dh_comms_dev.h:86` | Release exclusive sub-buffer access |
| `wave_signal_host(flags, idx, lane)` | `dh_comms_dev.h:104` | Signal host when buffer full, spin-wait until host clears |
| `generic_submit_message(...)` | `dh_comms_dev.h:115` | Common submit implementation used by all submit variants |

## Data Flow

1. Caller (GPU kernel) invokes a `v_submit_*` or `s_submit_*` function.
2. `get_sub_buffer_idx()` computes target sub-buffer from flattened workgroup index.
3. `wave_acquire()` atomically acquires exclusive write access to the sub-buffer.
4. `generic_submit_message()` writes wave_header, optional lane_headers, and data payload in dword (4-byte) units.
5. If the sub-buffer is full, `wave_signal_host()` sets the host-device atomic flag and spins until the host clears it.
6. `wave_release()` releases the lock.

## Invariants

- One wave writes at a time per sub-buffer (atomic acquire/release on `atomic_flags_d_`).
- Sub-buffer selection: flattened workgroup index modulo `no_sub_buffers`.
- Messages written in dword units (4 bytes), coalesced across lanes for vector submits.
- If buffer full, device signals host and spins until cleared -- this is a blocking operation.
- Lane header size assumed to be 4 bytes (optimized path); larger headers trigger slower calculation.

## Dependencies

- HIP runtime intrinsics: `__clock64`, `__builtin_amdgcn_read_exec`, `__ballot`, `__shfl`, etc.
- `dh_comms_descriptor` struct (passed as kernel argument via `get_dev_rsrc_ptr()`).

## Negative Knowledge

- Lane header size optimization only works for 4-byte headers. Larger headers fall back to a slower path.
- No support for non-AMD GPU architectures (relies on AMDGCN builtins).

## Open Questions

- Could lane header size flexibility be improved without sacrificing the fast path?

## Last Verified

- 2026-03-02 (migrated from KT dossiers 2026-04-27)
