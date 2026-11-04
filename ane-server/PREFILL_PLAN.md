# Prefill optimization acceptance plan

Baseline: commit 4ecf7f2, v6 asset, q64, concurrency4, 16K context.
All edits, export, build, correctness and timing run on this Mac mini.
Keep the deployed binary/model intact until acceptance; stage in a separate worktree.

A: baseline runtime/asset.
B: KV-only q32/q64 graphs; q1 remains responsible for final logits.
C: descriptor-checked persistent prefill buffers, incremental valid-token KV writes,
   independent ownership per logical request, and no redundant host COW copies.
D: bounded round-robin prefill work between decode steps.

Correctness: cold and warm prefixes, ragged lengths, ring wrap, q32/q64 tails,
row reuse, mixed prefill/decode, cancellation and deadlines. Same-shape runtime
changes must be bitwise equal; changed compiled graphs require >=40dB SNR and
matching top1, plus finite KV/logits. Compare against the validated v6 artifact.

Performance: separate correctness from timing; compiled/warm execution, cold
unique prefixes, 1K/4K/8K inputs and128/256 outputs; concurrency1/4.
Use ablations and alternating repeats. Also measure mixed long-prefill/ongoing
decode latency and per-request output gaps. Reject compiler-contaminated runs.
Do not promote a scheduler budget without measured latency benefit and a
documented throughput tradeoff.

References:
- https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html
  Registered shared I/O avoids redundant copies. Our NDArray reuse is analogous,
  not a claim of QNN-style zero-copy or explicit ANE SRAM residency.
- https://docs.cloud.google.com/tpu/docs/performance-guide
  Reduce intermediate memory traffic and respect actual hardware layout.
  TPU tiling constants do not apply to ANE.
