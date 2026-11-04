# Batch 4 versus batch 8

Same v7 model artifact and server binary on the M6 16GB mini, q64 prefill,
16K total context, temperature0. Each case uses unique system prefixes and
one observation. All cases passed the compiler-activity check; these are
not p95 measurements or a completely isolated machine benchmark.

Input lengths are nominal. The chat template adds roughly64 tokens; server
usage, rather than the Python tokenizer estimate, is authoritative.
Decode rate is per request; end-to-end rate includes queue/prefill time and
uses the total completed output tokens divided by the whole burst duration.

| Input | Output | Concurrency | Aggregate end-to-end tokens/s | Mean TTFT (s) | Mean decode tokens/s per request |
|---|---:|---:|---:|---:|---:|
| 1024 | 128 | 4 | 58.8 | 4.18 | 28.5 |
| 1024 | 128 | 8 | 68.6 | 7.55 | 17.3 |
| 1024 | 256 | 4 | 82.8 | 3.89 | 30.4 |
| 1024 | 256 | 8 | 93.4 | 7.53 | 17.8 |
| 4096 | 128 | 4 | 28.0 | 14.08 | 30.2 |
| 4096 | 128 | 8 | 28.9 | 28.01 | 17.1 |
| 4096 | 256 | 4 | 45.2 | 14.13 | 30.2 |
| 4096 | 256 | 8 | 48.2 | 28.12 | 17.9 |
| 8192 | 128 | 4 | 15.8 | 27.85 | 28.8 |
| 8192 | 128 | 8 | 16.4 | 55.03 | 17.2 |
| 8192 | 256 | 4 | 28.2 | 27.84 | 30.3 |
| 8192 | 256 | 8 | 29.4 | 55.30 | 18.0 |

Default recommendation: concurrency4 for interactive translation. With256
output tokens, increasing to8 improves aggregate throughput about13% at1K,
7% at4K and4% at8K, while approximately doubling time to first token.
B8 is numerically validated and remains available for workloads that prefer
throughput over response latency. This sweep does not prove full ANE occupancy.

The deployed default uses the smaller v6 B4 artifact, avoiding an unused B8
entrypoint. The same-artifact comparison above controls for export differences;
the earlier v6 serving numbers are listed separately in README.md.
