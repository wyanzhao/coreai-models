# q64 persistent state integration

Base:3bb6a3b. Production44cc503/v8 stays online until acceptance.
All execution on Mac mini. Isolated branch experiment/stateful-q64.

1. Export a separate q64/B1 state entrypoint. Preserve readonly attention and
   appended-query masks; commit updates to four flattened multilayer state
   buffers after computing the same new KV outputs. Do not alter q1 decoding.
2. Allocate runtime state descriptors per request. Restore host prefix snapshots
   only at bank binding. No ordinary host-to-state write after a stateful call.
3. Route only full q64 blocks at 64-aligned cursors and ring offsets. Unaligned
   restored prefixes retain baseline functional chunk boundaries; do not
   realign them by changing FP16 attention chunking.
   q32/partial tails and final logits use validated functional graphs; invalidate
   or rebind banks on descriptor/cursor changes. Cancellation releases ownership.
4. Require >=40dB logits SNR and matching top1 against v8, exact same-graph
   controls, state engagement counts, cold/warm prefixes, ring wrap, row reuse,
   disconnection and batch4. Micrograph correctness is not full-model acceptance.
5. Alternate baseline/candidate HTTP runs with cold prefixes: PP1K/4K/8K,
   TG128/256, concurrency1/4 and mixed traffic. Keep warmup, three samples,
   compiler witnesses, raw data and rollback. Deploy only stable improvements.
