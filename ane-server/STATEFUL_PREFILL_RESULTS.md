# q64 stateful prefill: integrated, not enabled
2026-11-03, Mac mini M6, macOS 27.0.1 (26A434), 16 GiB.

## Decision
Keep production port 18083 on accepted v8 / 44cc503. The opt-in separate-state implementation passed correctness, but the measured O/S campaign did not establish a stable benefit. No deployment was made. All model export, compilation and execution occurred on the mini.

## Implementation
The q64 prefill wrapper updates four persistent rank-4 buffers for global K/V (3 layers, 512 channels, capacity 16384) and sliding K/V (12 layers, 256 channels, capacity 576). Runtime allocates from CoreAI state descriptors rather than assuming dense layout, binds mutable state, and retains returned KV for CPU prefix snapshots and decode rebinding.
State calls require a naturally aligned, complete q64 chunk; prefix restoration preserves original chunk boundaries. Unaligned continuations, partial tails and final q1 remain functional. Existing singleton q1 uses B2 with a dummy row. Failed calls discard the bank before host KV commit; canceled/reused slots do not inherit another request's bank.
Enable only with --stateful-prefill-model pointing to the separate versioned asset. Default remains off. Explicit mutable state is not proof of internal zero-copy inside the compiler.

## Correctness and engagement
- 30 tests in five suites passed. The full-model parity test compares 30 outputs bitwise: cold lengths 1 through 16384, prefixes restored at 64/95/1001/1151 tokens, ragged four-row decode, and abandoned-bank/row reuse.
- Correctness witness: 567 state calls, 36288 tokens, 23 bank creations/restores.
- Initial alignment-bridging variant failed one later decode comparison at 39.05 dB. Preserving original chunk boundaries fixed it; failing evidence is retained.
- HTTP checks passed authorization, output cap, prefix restoration, queue overload and cancellation/slot recovery. API counters witnessed 24 state calls / 1536 tokens.
- Separate state model placement: 2680 ANE / 2 GPU operations. Counts describe placement, not utilization or an ANE-only route.

## Performance
One warmup per matrix pair, three timed samples, nominal PP 1024/4096/8192, TG 128/256, concurrency 1/4, serial baseline then stateful. All 72 timed cases were compiler-quiet and reused zero prefix tokens. The 6280 state-call delta matches expected complete q64 chunks.
Inputs were randomized independently; template overhead means these are not exact 1K/4K/8K token prompts. This is a negative screening result, not a paired causal microbenchmark. No pure PP TPS is inferred from HTTP TTFT.

Concurrency 4, TG 256; throughput is aggregate output tokens divided by entire HTTP batch wall time:

| Nominal PP | Actual prompt range baseline / state | Baseline tok/s | State tok/s | Baseline mean TTFT s | State mean TTFT s |
|---:|---|---:|---:|---:|---:|
| 1024 | 1084–1091 / 1084–1091 | 103.47 | 102.06 | 1.207 | 1.279 |
| 4096 | 4156–4163 / 4157–4164 | 81.98 | 80.87 | 3.797 | 3.876 |
| 8192 | 8253–8260 / 8252–8260 | 64.60 | 63.77 | 7.345 | 7.285 |

Stateful serving used roughly 300–450 MiB more resident memory in these measurements. Full grouped metrics, including per-request decode rate, are in stateful-prefill-results.json. No stable speedup was found; do not enable based on operation placement or the mutable-state API alone.

## Follow-up experiment retained, not admitted
A combined five-entrypoint asset (~936 MiB) exported and specialized (19660 ANE / 12 GPU operations), intended to avoid a second model instance. Its parity run failed before inference with CoreAI failedToPurge("Deletion could not be completed, assets still in use"). Serial retries, canonical paths and an APFS-cloned v8 control did not resolve it. A standalone public persistent-cache specialization could load v8 q64, but did not fix the normal test loader. Root cause remains unproven; no production cache was manually purged.
Combined-asset runtime/export edits and deterministic paired benchmark seeds are preserved only in results/combined-state-and-paired-benchmark-wip.patch; they are not part of the validated implementation or deployment.

## Evidence on the mini
Worktree: /Users/yzwang/Workspace/coreai-native-kv
- ane-server/results/stateful-final-tests.log
- ane-server/results/stateful-parity-v1.log and stateful-v1-source.patch
- ane-server/results/http-stateful-metrics.json
- ane-server/results/stateful-matrix-os/ (identity, all raw requests, server/probe logs, counters)
- ane-server/results/stateful-combined-parity-*.log and cache-probe-v8.log
- Assets under /Users/yzwang/Workspace/coreai-models/ane-assets/
Measured implementation tree: 489c9b74cca6fc3cba8884c7145208c8f11f2f1b; binary SHA256 is recorded in the JSON.
