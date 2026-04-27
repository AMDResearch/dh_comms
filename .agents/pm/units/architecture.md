# Architecture

## Responsibility

Device-host communication library for streaming messages from GPU kernels to host code. Provides shared buffer infrastructure, synchronization primitives, and a message handler framework.

## Key Source Files

| File | Role |
|------|------|
| `include/dh_comms_dev.h` | Device-side API: `v_submit_*`, `s_submit_*` functions |
| `include/dh_comms.h` | Host-side API: `dh_comms` class, resource management |
| `src/dh_comms.cpp` | Host-side implementation: buffer allocation, polling, handler dispatch |
| `include/message.h` | `message_t` type and type enums |
| `include/message_handlers.h` | `message_handler_base` abstract class and handler chain |
| `include/data_headers.h` | Host-side wire format structs (wave_header, lane_header) |
| `include/data_headers_dev.h` | Device-side wire format structs |
| `src/data_headers.cpp` | Wire format implementation |
| `src/message.cpp` | Message parsing implementation |
| `src/message_handlers.cpp` | Handler chain implementation |
| `include/gpu_arch_constants.h` | GPU architecture constants (wave size, etc.) |
| `include/hip_utils.h` | HIP runtime utility wrappers |
| `include/utils.h` | General utilities |

## Key Types and Classes

| Type | File | Role |
|------|------|------|
| `dh_comms` | `include/dh_comms.h` | Main orchestrator: allocates buffers, runs polling thread, dispatches to handler chain |
| `dh_comms_resources` | `include/dh_comms.h` | Buffer allocation container |
| `dh_comms_descriptor` | `include/dh_comms.h` | Struct passed to device code containing buffer pointers and sizes |
| `dh_comms_mem_mgr` | `include/dh_comms.h` | Memory allocator abstraction (default: HIP coherent memory) |
| `message_t` | `include/message.h` | Parsed message representation |
| `message_handler_base` | `include/message_handlers.h` | Abstract base class for message handlers |
| `message_handler_chain_t` | `include/message_handlers.h` | Ordered chain of handlers |
| `wave_header_t` | `include/data_headers.h` | Per-wave metadata (exec mask, timestamp, DWARF info) |
| `lane_header_t` | `include/data_headers.h` | Per-lane metadata (thread coordinates) |

## Key Functions and Entry Points

| Function | File | Role |
|----------|------|------|
| `v_submit_message()` | `dh_comms_dev.h:255` | Vector submit: all active lanes submit data |
| `s_submit_message()` | `dh_comms_dev.h:279` | Scalar submit: first active lane submits data |
| `v_submit_address()` | `dh_comms_dev.h:318` | Specialized address message submit |
| `s_submit_wave_header()` | `dh_comms_dev.h:304` | Header-only submit, no data payload |
| `s_submit_time_interval()` | `dh_comms_dev.h:333` | Time interval message submit |
| `get_sub_buffer_idx()` | `dh_comms_dev.h:36` | Compute sub-buffer index for current wave |
| `wave_acquire()` | `dh_comms_dev.h:65` | Acquire exclusive sub-buffer access |
| `wave_release()` | `dh_comms_dev.h:86` | Release sub-buffer access |
| `wave_signal_host()` | `dh_comms_dev.h:104` | Signal host when buffer full, spin until cleared |
| `generic_submit_message()` | `dh_comms_dev.h:115` | Common submit implementation |
| `dh_comms::start()` | `dh_comms.h:99` | Launch processing thread |
| `dh_comms::stop()` | `dh_comms.h:101` | Stop processing, join thread, drain remaining messages |
| `dh_comms::get_dev_rsrc_ptr()` | `dh_comms.h:97` | Get device-visible descriptor for kernel arg |
| `processing_loop()` | `src/dh_comms.cpp` | Host polling thread: polls flags, parses messages, invokes handlers |

## Data Flow

1. Kernel calls `v_submit_message()` with message data and DWARF info.
2. Device code selects sub-buffer based on flattened workgroup index modulo number of sub-buffers.
3. Wave acquires exclusive access via `atomic_flags_d_`.
4. Wave writes header plus lane data to buffer.
5. If buffer full: signal host via `atomic_flags_hd_`, spin until cleared.
6. Wave releases lock.
7. Host polling thread sees flag, parses messages, invokes handler chain.
8. Host clears flag, device resumes.

Shared memory uses `hipHostMallocCoherent` for host-device visibility.

## Invariants

- Sub-buffers provide parallelism; waves writing to the same sub-buffer serialize via atomic acquire/release.
- Host-device synchronization uses coherent memory (`hipHostMallocCoherent`).
- Message format: `wave_header` followed by optional `lane_header` entries followed by data items.
- Handler chain: first handler returning `true` consumes the message (unless pass-through mode enabled).
- `start()` must be called before kernel dispatch; `stop()` must be called after kernel completion.

## Dependencies

- HIP runtime (memory allocation, device intrinsics)
- CMake build system
- Produces static/shared library linked into consumer projects (e.g., omniprobe)

## Negative Knowledge

- No parallel sub-buffer processing on the host side -- single polling thread only.
- No parallel handler invocation within the chain.
- Default `dh_comms_mem_mgr` uses HIP; custom manager needed for HSA-only contexts.

## Open Questions

- Could parallel sub-buffer processing improve throughput for high-bandwidth workloads?
- Is the single-thread polling design a bottleneck for large sub-buffer counts?

## Last Verified

- 2026-03-02 (migrated from KT dossiers 2026-04-27)
