# dh_comms Architecture

## Overview
Device-host communication library for streaming messages from GPU kernels to host code. Provides shared buffer infrastructure, synchronization primitives, and a message handler framework.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GPU Kernel                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  v_submit_message() / s_submit_message() / v_submit_address()│   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │  Sub-buffer selection (workgroup-based)                      │   │
│  │  wave_acquire() → write → wave_release()                     │   │
│  │  wave_signal_host() if buffer full                           │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────────────┘
                              │ Shared Memory (hipHostMallocCoherent)
┌─────────────────────────────▼───────────────────────────────────────┐
│                          Host Thread                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  processing_loop() polls atomic_flags_hd_                    │   │
│  │  → parse messages → invoke handler chain                     │   │
│  │  → clear sub-buffer → reset flag                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │  message_handler_chain_t                                     │   │
│  │  → handler1.handle() → handler2.handle() → ...              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. Kernel calls `v_submit_message()` with message data + DWARF info
2. Device code selects sub-buffer based on flattened workgroup index
3. Wave acquires exclusive access via `atomic_flags_d_`
4. Wave writes header + lane data to buffer
5. If buffer full: signal host via `atomic_flags_hd_`, spin until cleared
6. Wave releases lock
7. Host polling thread sees flag, parses messages, invokes handlers
8. Host clears flag, device resumes

## Key Invariants

- Sub-buffers provide parallelism; waves writing to same sub-buffer serialize
- Host-device synchronization uses coherent memory (hipHostMallocCoherent)
- Message format: wave_header + optional lane_headers + data items
- Handler chain: first handler returning true "consumes" the message

## Subsystems

| Component | Files | Description |
|-----------|-------|-------------|
| Device API | `include/dh_comms_dev.h` | `v_submit_*`, `s_submit_*` device functions |
| Host API | `include/dh_comms.h`, `src/dh_comms.cpp` | `dh_comms` class, resource management |
| Message Types | `include/message.h` | `message_t`, type enums |
| Handlers | `include/message_handlers.h` | `message_handler_base`, chain |
| Data Headers | `include/data_headers.h`, `include/data_headers_dev.h` | wire formats |

## Build

- CMake-based, requires HIP runtime
- Produces: static/shared library linked into consumers

## Last Verified
Date: 2026-03-02
