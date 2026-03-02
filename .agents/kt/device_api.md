# Device API

## Responsibility
Device-side functions for submitting messages from GPU kernels to shared buffers accessible by host code.

## Core Concepts
- **Vector Message**: All active lanes submit data (e.g., 64 addresses)
- **Scalar Message**: Only first active lane submits (e.g., timestamps)
- **Wave Header**: Per-wave metadata (exec mask, timestamp, DWARF info)
- **Lane Header**: Per-lane metadata (thread coordinates)
- **Sub-buffer**: Partition of main buffer; waves distribute across sub-buffers

## Key Invariants
- One wave writes at a time per sub-buffer (atomic acquire/release)
- Sub-buffer selection based on flattened workgroup index % no_sub_buffers
- Messages written in dword units (4 bytes), coalesced across lanes
- If buffer full, device signals host and spins until cleared

## Interfaces

### Primary Functions
- `v_submit_message(rsrc, msg, size, ...)` — vector submit, all lanes — `dh_comms_dev.h:255`
- `s_submit_message(rsrc, msg, size, ...)` — scalar submit, first lane — `dh_comms_dev.h:279`
- `v_submit_address(rsrc, addr, ...)` — specialized for address messages — `dh_comms_dev.h:318`
- `s_submit_wave_header(rsrc, ...)` — header only, no data — `dh_comms_dev.h:304`
- `s_submit_time_interval(rsrc, ...)` — time interval message — `dh_comms_dev.h:333`

### Internal Functions
- `get_sub_buffer_idx(no_sub_buffers)` — compute sub-buffer for wave — `dh_comms_dev.h:36`
- `wave_acquire(flags, idx, lane)` — acquire exclusive access — `dh_comms_dev.h:65`
- `wave_release(flags, idx, lane)` — release access — `dh_comms_dev.h:86`
- `wave_signal_host(flags, idx, lane)` — signal host, wait for clear — `dh_comms_dev.h:104`
- `generic_submit_message(...)` — common implementation — `dh_comms_dev.h:115`

## Message Format

```
┌─────────────────────────────────────────┐
│ wave_header_t                           │
│  - exec_mask (64 bits)                  │
│  - data_size                            │
│  - timestamp                            │
│  - dwarf_fname_hash, line, column       │
│  - user_type, user_data                 │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ lane_header_t × active_lane_count       │
│  (optional, contains thread coords)     │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ data × submitting_lane_count            │
│  (message payload, dword-aligned)       │
└─────────────────────────────────────────┘
```

## Address Message Encoding

`v_submit_address()` packs metadata into `user_data`:
- Bits 0-1: rw_kind (read=1, write=2, read_write=3)
- Bits 2-5: memory_space (flat=0, global=1, shared=3, etc.)
- Bits 6+: sizeof_pointee

## Dependencies
- HIP runtime intrinsics (`__clock64`, `__builtin_amdgcn_read_exec`, etc.)

## Known Limitations
- Lane header size assumed 4 bytes (optimized path); larger headers use slower calculation

## Last Verified
Date: 2026-03-02
