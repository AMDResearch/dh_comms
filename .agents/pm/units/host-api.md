# Host API

## Responsibility

Host-side management of shared buffers and message processing. Allocates coherent GPU-visible memory, runs a polling thread to receive messages from the device, and dispatches received messages to the handler chain.

## Key Source Files

| File | Role |
|------|------|
| `include/dh_comms.h` | `dh_comms` class, `dh_comms_resources`, `dh_comms_descriptor`, `dh_comms_mem_mgr` |
| `src/dh_comms.cpp` | Implementation: buffer allocation, polling loop, handler dispatch |

## Key Types and Classes

| Type | File | Role |
|------|------|------|
| `dh_comms` | `dh_comms.h` | Main orchestrator class |
| `dh_comms_resources` | `dh_comms.h` | Buffer allocation container |
| `dh_comms_descriptor` | `dh_comms.h` | Struct passed to device code (pointers + sizes) |
| `dh_comms_mem_mgr` | `dh_comms.h` | Memory allocator abstraction (default uses HIP coherent memory) |

### dh_comms_descriptor Layout

```cpp
struct dh_comms_descriptor {
  size_t no_sub_buffers_;
  size_t sub_buffer_capacity_;
  char *buffer_;
  size_t *sub_buffer_sizes_;
  uint32_t *error_bits_;
  uint8_t *atomic_flags_d_;   // device-only sync
  uint8_t *atomic_flags_hd_;  // host-device sync
};
```

## Key Functions and Entry Points

| Function | Location | Role |
|----------|----------|------|
| `dh_comms(no_sub_buffers, capacity, kdb, verbose, ...)` | `dh_comms.h:85` | Constructor: allocates buffers via `dh_comms_mem_mgr` |
| `get_dev_rsrc_ptr()` | `dh_comms.h:97` | Returns pointer to device-visible descriptor for kernel arg |
| `start()` / `start(kernel_name)` | `dh_comms.h:99-100` | Launch `sub_buffer_processor_` polling thread |
| `stop()` | `dh_comms.h:101` | Set `running_ = false`, join thread, run final processing loop to drain |
| `append_handler(handler)` | `dh_comms.h:106` | Add handler to chain |
| `report(auto_clear)` | `dh_comms.h:117` | Invoke `report()` on all handlers |
| `clear_handler_states()` | `dh_comms.h:113` | Reset handlers for reuse across kernel dispatches |
| `delete_handlers()` | `dh_comms.h:115` | Remove all handlers |
| `dh_comms_mem_mgr::calloc(size)` | `dh_comms.h:40` | Allocate host-coherent memory |
| `dh_comms_mem_mgr::calloc_device_memory(size)` | `dh_comms.h:41` | Allocate device memory |
| `dh_comms_mem_mgr::free()` / `free_device_memory()` | `dh_comms.h:42-43` | Deallocate memory |
| `dh_comms_mem_mgr::copy_to_device()` | `dh_comms.h:45` | Host-to-device copy |

## Data Flow

1. Constructor allocates buffers via `dh_comms_mem_mgr` (coherent memory).
2. `get_dev_rsrc_ptr()` returns a pointer to the `dh_comms_descriptor` for passing as a kernel argument.
3. `start()` launches the `sub_buffer_processor_` thread.
4. Polling thread iterates sub-buffers in round-robin, checking `atomic_flags_hd_`.
5. When a flag is set: parse messages from that sub-buffer, invoke handler chain.
6. Clear the flag so the device can resume writing.
7. `stop()` sets `running_ = false`, joins the thread, and runs a final processing loop to drain any remaining messages.
8. `report()` invokes `report()` on all handlers in the chain.

## Invariants

- `start()` must be called before kernel dispatch.
- `stop()` must be called after kernel completion (blocks until all messages processed).
- Handler chain is invoked in order; first match wins (unless pass-through).
- Resources are freed in the destructor after all processing is complete.
- Single processing thread (`sub_buffer_processor_`), polls all sub-buffers in round-robin.
- Final processing loop after `stop()` drains remaining messages.

## Dependencies

- HIP runtime (memory allocation via `hipHostMallocCoherent`).
- `message_handler_base` / `message_handler_chain_t` (handler interface).
- `kernelDB` (optional, for ISA correlation).

## Negative Knowledge

- Single processing thread only -- no parallel sub-buffer processing.
- Default `dh_comms_mem_mgr` uses HIP; a custom memory manager is needed for HSA-only contexts.
- `stop()` blocks the calling thread until the processing thread joins and remaining messages are drained.

## Open Questions

- Would multiple processing threads (one per sub-buffer or per sub-buffer group) improve latency?
- Is there a need for an HSA-native memory manager implementation?

## Last Verified

- 2026-03-02 (migrated from KT dossiers 2026-04-27)
