# Host API

## Responsibility
Host-side management of shared buffers and message processing. Allocates resources, runs polling thread, invokes message handlers.

## Core Concepts
- **dh_comms**: Main orchestrator class
- **dh_comms_resources**: Buffer allocation container
- **dh_comms_descriptor**: Struct passed to device code (pointers + sizes)
- **dh_comms_mem_mgr**: Memory allocator abstraction (default uses HIP)

## Key Invariants
- `start()` must be called before kernel dispatch
- `stop()` must be called after kernel completion (blocks until messages processed)
- Handler chain invoked in order; first match wins
- Resources freed in destructor after all processing complete

## Data Flow

1. Constructor allocates buffers via `dh_comms_mem_mgr`
2. `get_dev_rsrc_ptr()` returns pointer to device-visible descriptor
3. `start()` launches `sub_buffer_processor_` thread
4. Thread polls `atomic_flags_hd_`, processes messages when flag set
5. `stop()` sets `running_ = false`, joins thread, calls final processing loop
6. `report()` invokes `report()` on all handlers

## Interfaces

### dh_comms Class
- `dh_comms(no_sub_buffers, capacity, kdb, verbose, ...)` — constructor — `dh_comms.h:85`
- `get_dev_rsrc_ptr()` — get descriptor for kernel arg — `dh_comms.h:97`
- `start()` / `start(kernel_name)` — begin processing — `dh_comms.h:99-100`
- `stop()` — end processing, join thread — `dh_comms.h:101`
- `append_handler(handler)` — add to chain — `dh_comms.h:106`
- `report(auto_clear)` — invoke handler reports — `dh_comms.h:117`
- `clear_handler_states()` — reset handlers for reuse — `dh_comms.h:113`
- `delete_handlers()` — remove all handlers — `dh_comms.h:115`

### dh_comms_mem_mgr Class
- `calloc(size)` — allocate host-coherent memory — `dh_comms.h:40`
- `calloc_device_memory(size)` — allocate device memory — `dh_comms.h:41`
- `free()` / `free_device_memory()` — deallocation — `dh_comms.h:42-43`
- `copy_to_device()` — H2D copy — `dh_comms.h:45`

### dh_comms_descriptor Struct
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

## Dependencies
- HIP runtime (memory allocation)
- message_handler_base (handler interface)
- kernelDB (optional, for ISA correlation)

## Threading Model
- Single processing thread (`sub_buffer_processor_`)
- Polls all sub-buffers in round-robin
- Final processing loop after `stop()` drains remaining messages

## Known Limitations
- Single processing thread (no parallel sub-buffer processing)
- Default mem_mgr uses HIP; custom mgr needed for HSA-only contexts

## Last Verified
Date: 2026-03-02
