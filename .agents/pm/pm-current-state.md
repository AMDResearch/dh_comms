# Project Memory Current State

## Active Work Areas

- `dh_comms` is a device-host communication library for streaming messages from GPU kernels to host code on AMD GPUs.
- Primary type: `code`. Enabled facets: code, design.
- The library provides shared buffer infrastructure, synchronization primitives, and a message handler framework.

## Current Risks

- PM was migrated from KT dossiers dated 2026-03-02. Source code may have evolved since then; re-verify units against current source on first substantial task.
- Single-thread polling design may be a performance bottleneck for high-throughput workloads.

## Active Workflows

- None yet.

## Recently Changed Assumptions

- KT system migrated to v0.3 PM on 2026-04-27. KT originals archived in `.agents/kt.archive/`.

## Recommended Reading Order For A Fresh Session

1. `.agents/pm/pm-index.md`
2. `.agents/state/current-focus.md`
3. `.agents/state/active-workflows.md`
4. `.agents/pm/units/architecture.md`
5. `.agents/pm/units/device-api.md` (if working on device-side code)
6. `.agents/pm/units/host-api.md` (if working on host-side code)
7. `.agents/pm/units/handlers.md` (if working on message handlers)
