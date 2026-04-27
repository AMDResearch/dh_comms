# Handlers

## Responsibility

Host-side framework for processing messages received from GPU kernels. Provides an abstract handler base class and an ordered chain mechanism for routing messages to appropriate handlers.

## Key Source Files

| File | Role |
|------|------|
| `include/message_handlers.h` | `message_handler_base` abstract class and `message_handler_chain_t` |
| `src/message_handlers.cpp` | Handler chain implementation |
| `include/message.h` | `message_t` type used by handlers |
| `src/message.cpp` | Message parsing implementation |

## Key Types and Classes

| Type | File | Role |
|------|------|------|
| `message_handler_base` | `message_handlers.h` | Abstract base class: defines `handle()`, `report()`, `clear()` interface |
| `message_handler_chain_t` | `message_handlers.h` | Ordered list of handlers; dispatches messages through the chain |
| `message_t` | `message.h` | Parsed message representation passed to handlers |
| `message_type` | `message.h` | Enum: `address` (0), `time_interval` (1), `basic_block_start` (2) |

## Key Functions and Entry Points

| Function | Location | Role |
|----------|----------|------|
| `message_handler_base::handle(message)` | `message_handlers.h:49` | Process a message; return `true` if consumed |
| `message_handler_base::handle(message, kernel, kdb)` | `message_handlers.h:50` | Process with ISA context for correlation |
| `message_handler_base::report()` | `message_handlers.h:56` | Output accumulated results |
| `message_handler_base::report(kernel_name, kdb)` | `message_handlers.h:57` | Output with ISA context |
| `message_handler_base::clear()` | `message_handlers.h:61` | Reset state for handler reuse |
| `message_handler_chain_t::add_handler(handler)` | `message_handlers.h:75` | Append handler to chain |
| `message_handler_chain_t::handle(message)` | `message_handlers.h:73` | Invoke chain: pass message to each handler until consumed |
| `message_handler_chain_t::report()` | `message_handlers.h:76` | Call `report()` on all handlers |
| `message_handler_chain_t::clear_handler_states()` | `message_handlers.h:78` | Clear state on all handlers |
| `message_handler_chain_t::clear()` | `message_handlers.h:79` | Remove all handlers from chain |

## Data Flow

1. Handler is constructed (by plugin or caller).
2. `add_handler()` appends it to the chain.
3. Kernel runs, messages stream from device to host.
4. `handle()` is called per message, passing through the chain in order.
5. First handler returning `true` consumes the message (unless pass-through mode is enabled).
6. After kernel completes, `stop()` is called.
7. `report()` is called on all handlers to output accumulated results.
8. `clear()` resets state or destroys handlers.

## Invariants

- `handle()` returns `true` if the message is consumed, `false` to pass to next handler.
- Pass-through mode: if enabled, message continues to next handler even after a match.
- Stateful handlers accumulate data across messages; stateless handlers process on-the-fly.
- `report()` is called after all messages are processed.
- `clear()` resets state for handler reuse across multiple kernel dispatches.
- No parallel handler invocation -- handlers are called sequentially.

## Dependencies

- `message.h` / `message_t` for the parsed message type.
- `kernelDB` (optional) for ISA-aware handling via the two-argument `handle()` and `report()` overloads.

## Negative Knowledge

- No parallel handler invocation within the chain.
- Handler must be thread-safe if `dh_comms` is used from multiple host threads.
- The chain does not support handler removal by reference; `clear()` removes all.

## Open Questions

- Would a priority-based handler dispatch model be useful for complex handler sets?
- Should the framework provide built-in thread safety for multi-threaded host usage?

## Last Verified

- 2026-03-02 (migrated from KT dossiers 2026-04-27)
