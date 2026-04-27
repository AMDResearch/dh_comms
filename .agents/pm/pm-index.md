# Project Memory Index

Project memory units record durable, reusable project knowledge. Load only the units relevant to the current task.

| Unit | Purpose | Status | Facet | When To Load | Dependencies |
|------|---------|--------|-------|--------------|--------------|
| `architecture` | Top-level system diagram, data flow, subsystem map, and build overview for dh_comms. | verified | code | Load at session start or when the task spans multiple subsystems. | None (root unit) |
| `device-api` | Device-side submit functions, sub-buffer management, synchronization primitives, and message wire format. | verified | code | Load before changing device-side code in `include/dh_comms_dev.h` or `include/data_headers_dev.h`. | architecture |
| `handlers` | Host-side message handler framework: abstract base class, handler chain, lifecycle, and predefined message types. | verified | code | Load before changing handler code in `include/message_handlers.h` or `src/message_handlers.cpp`. | architecture |
| `host-api` | Host-side buffer management, polling thread, descriptor struct, memory manager, and start/stop lifecycle. | verified | code | Load before changing host-side code in `include/dh_comms.h` or `src/dh_comms.cpp`. | architecture, handlers |

## Usage Notes

- Keep PM selective. Do not turn it into a transcript dump.
- Split units when they stop being task-oriented.
- Move completed historical units into `.agents/pm/done/` when appropriate.
- The `architecture` unit provides the top-level overview; load it first when working across subsystems.
- The `device-api`, `handlers`, and `host-api` units are task-specific; load only what is needed.
