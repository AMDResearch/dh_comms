# Message Handlers

## Responsibility
Framework for processing messages on the host. Provides abstract base class and chain mechanism for routing messages to appropriate handlers.

## Core Concepts
- **Handler Chain**: Ordered list of handlers; message passed to each until one returns true
- **Pass-through Mode**: If enabled, message continues to next handler even after match
- **Stateful vs Stateless**: Stateful handlers accumulate data; stateless process on-the-fly

## Key Invariants
- `handle()` returns true if message consumed, false to pass to next handler
- `report()` called after all messages processed
- `clear()` resets state for handler reuse

## Interfaces

### message_handler_base (Abstract)
- `handle(message)` — process message, return true if handled — `message_handlers.h:49`
- `handle(message, kernel, kdb)` — with ISA context — `message_handlers.h:50`
- `report()` — output accumulated results — `message_handlers.h:56`
- `report(kernel_name, kdb)` — with ISA context — `message_handlers.h:57`
- `clear()` — reset state — `message_handlers.h:61`

### message_handler_chain_t
- `add_handler(handler)` — append to chain — `message_handlers.h:75`
- `handle(message)` — invoke chain — `message_handlers.h:73`
- `report()` — call report on all handlers — `message_handlers.h:76`
- `clear_handler_states()` — clear all handlers — `message_handlers.h:78`
- `clear()` — remove all handlers — `message_handlers.h:79`

## Handler Lifecycle

```
1. Construction (by plugin or caller)
2. append_handler() → added to chain
3. Kernel runs, messages stream
4. handle() called per message
5. stop() called
6. report() called
7. clear() or destruction
```

## Implementing a Handler

```cpp
class my_handler : public dh_comms::message_handler_base {
public:
  bool handle(const message_t& msg) override {
    if (msg.wave_header().user_type_ != my_type) return false;
    // process message
    return true;
  }
  void report() override { /* output results */ }
  void clear() override { /* reset state */ }
};
```

## Message Types (Predefined)
- `message_type::address` (0) — memory address data
- `message_type::time_interval` (1) — timing data
- `message_type::basic_block_start` (2) — basic block entry

## Dependencies
- message.h (message_t class)
- kernelDB (optional, for ISA-aware handling)

## Known Limitations
- No parallel handler invocation
- Handler must be thread-safe if dh_comms used from multiple threads

## Last Verified
Date: 2026-03-02
