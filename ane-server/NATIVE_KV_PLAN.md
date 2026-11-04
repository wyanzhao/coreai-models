# Native KV acceptance plan

Baseline:44cc503, v8 asset, batch4, q64, budget1, 16K context.
All work and measurements on Mac mini; serving ports18081/18082/18083 preserved.
1. Direct runtime-layout KV outputs to host snapshots and persistent input buffers.
   Keep a legacy output-materialization flag for bitwise and timing ablation.
2. Probe explicit mutable states, independent row offsets, inactive rows and
   repeated-call persistence using installed CoreAI APIs; inspect placement.
3. After probe evidence, prioritize aligned q64 prefill; keep q1 decode functional.
   Integration requires valid-token,
   ring-wrap, prefix-isolation and row-reuse checks. Reject GPU fallback or copies
   that erase the benefit; do not assume matching shapes imply matching layout.
4. Same-graph runtime changes require bitwise parity. Changed graphs require
   finite outputs, >=40dB SNR and matching top1. Test ragged rows and prefix reuse.
5. Measure warm compiler-quiet alternating comparisons, standalone decode and
   mixed prefill/decode. Promote only demonstrated improvements; keep rollback.

## State boundary established on November 3

- B1 rank5 fused mutable cache, descriptor-backed IOSurface/interleave8:
  q1 offsets0/4/8/60 and repeated0 pass; offset1 fails on ANE. GPU
  offsets0/1/2/63 pass. This is specific to these graphs/runtime, not a
  general ANE alignment specification.
- q64/C512/capacity16384 offsets0/64/128/16320 pass exact output/state.
- q64/C256/ring576 offsets0/512/0/64 pass exact output/state.
- Both q64 probe compilations report9 ANE and3 GPU residency entries. The
  update source maps to ANE, but these counts do not establish utilization,
  zero-copy execution, or the absence of state-copy/bridging overhead.
- Full-model mutable-state integration remains gated on multi-layer physical
  layout compatibility, attention parity, prefix restoration at arbitrary
  cursors, partial tail handling, cancellation/reuse and measured benefit.
  Align a restored cursor using the existing functional path before entering
  q64 stateful prefill; do not pad over committed prefix tokens. A shared
  runtime must retain per-request ownership of mutable state.
- The native-output experiment still keeps CPU snapshots for prefix/rebind;
  it is not an end-to-end device-native KV cache.
