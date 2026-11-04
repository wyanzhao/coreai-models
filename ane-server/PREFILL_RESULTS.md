# Prefill optimization results

Mac mini, Apple M6, 16GiB, macOS27.0.1 (26A434). All exports, builds,
correctness checks and measurements ran on this Mac. Source weights:
google/gemma-4-E2B-it revision4adf15748de9e1a0097b6274cee9c15836c41c37.

The accepted candidate uses KV-only intermediate prefill, persistent per-request
KV buffers, and one round-robin prefill chunk between decode steps.
Decode capacity remains4 and total context remains16384.

## Controlled comparisons

A: v6 asset, old prefill packing, visit all prefilling rows per step.
B: v8 KV-only intermediate prefill, old packing, visit all rows.
C: v8, persistent KV, visit all rows.
D: v8, persistent KV, globally budget one chunk per step.

Sequence: A,D,D,A,B,C,C,B, with a new server process and warmup per run.
A/D cover nominal1K/4K/8K inputs,128/256 outputs,concurrency1/4.
B/C cover4K inputs,128/256 outputs,concurrency1/4.
All64 measured sweep observations had zero restored prefix tokens, matched
the requested output count, and passed the compiler-quiet check.

Means of two observations, concurrency4,256 output tokens per request:

| Nominal input | Baseline E2E tokens/s | Candidate E2E tokens/s | Gain | Baseline mean TTFT | Candidate mean TTFT |
|---|---:|---:|---:|---:|---:|
| 1K | 84.15 | 105.63 | 25.5% | 3.808s | 1.206s |
| 4K | 46.03 | 83.23 | 80.8% | 14.004s | 3.815s |
| 8K | 28.46 | 64.72 | 127.4% | 27.610s | 7.332s |

Single-request decode remains approximately48tokens/s; the principal gain
is prefill. Prompt labels are nominal; actual server usage is retained in
[prefill-results.json](prefill-results.json), together with ranges and binary
identity. Two observations are not population p95 latency estimates.

At4K/concurrency4/output256, the ablation is:

| Variant | Mean TTFT | Aggregate E2E tokens/s |
|---|---:|---:|
| A: baseline | 14.004s | 46.03 |
| B: only remove intermediate logits | 6.736s | 67.70 |
| C: add persistent prefill KV | 3.725s | 85.74 |
| D: add global prefill budget1 | 3.815s | 83.23 |

Budget1 costs2.9% of this pure-throughput case relative toC. It improves
interactive decoding when new long prompts arrive.

## Mixed long/short requests

Start one1K request with256 output tokens. After its first output token,
submit three8K requests with128 outputs each. Values below use only
compiler-quiet-instrumented runs; the first A run predates that instrumentation
and remains in the raw summary but is excluded here.

| Variant | Short request total time | Short request within-stream p95 token gap |
|---|---:|---:|
| A | 27.89s | 179ms |
| C | 12.37s (two-run mean) | 62ms |
| D | 9.04s (two-run mean) | 36ms |

D's longest token gaps were57–68ms; the instrumented A run had329ms.
These are gaps within measured streams, not p95 request latency across users.
For translation, D is the default because its small throughput tradeoff buys
substantially smoother progress while long paragraphs enter the service.

## Correctness and traffic evidence

- 25 final-logit comparisons per candidate variant, spanning32,65,577,600
  and1089 prompt tokens, three decode steps and prefix restoration.
  Headless versus baseline and persistent versus nonpersistent results were
  all bitwise identical, finite, and had matching top1.
- Existing real-model checks passed: independent full-width reference,
  ragged B2/B4 decode, ring wrap, cross-request prefix restoration and q32/q64
  chunking. The pre-established40dB plus matching-top1 criterion remains
  unchanged for different chunk boundaries.
- Scheduler/API/KV/plan checks passed, including the global chunk budget.
- For the25-comparison fixture, the logical payload packed for multi-token
  prefill KV fell from4,094,164,992bytes to0; incremental KV writes totalled
  43,462,656bytes. Logits readback, including the required final/decode logits,
  fell from1,267,728,384 to26,214,400bytes.
  These counters are application payloads, not measured DRAM traffic or
  proof of zero-copy/ANE SRAM residency. The q1 tail still uses full-cache
  preparation, and prefix snapshot COW copies remain necessary for isolation.

## Vendor techniques and limits

The [Qualcomm HTP guidance](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/htp_backend.html)
motivates avoiding repeated I/O preparation through reusable buffers.
The [TPU performance guide](https://docs.cloud.google.com/tpu/docs/performance-guide)
motivates reducing intermediate tensor traffic and respecting physical layout.
This implementation checks CoreAI's actual NDArray descriptors; it does not
copy QNN allocation APIs or TPU tile dimensions onto ANE.

The first C4K/output128/single-request measurement had a decode slowdown that
did not recur. It is retained in the results rather than discarded. Compiler
quiet does not guarantee complete system isolation. Placement operation counts
do not establish runtime occupancy or utilization.

The scripts benchmark_server.py, benchmark_mixed.py and
run_prefill_comparison.py reproduce the measurements. The latter expects the
named v6/v8 assets under ane-assets and a bearer-token file supplied with
--api-key-file; it uses a dedicated localhost test port18084.

## Compiled placement

The v8 debug specialization reports16,980 ANE-resident and10 GPU-resident
optimized operations, compared with23,416/12 for v6. The compressed diagnostic
archive was decompressed and SHA256-verified. This confirms an ANE-dominant
compiled route; operation counts are not weighted timing or ANE occupancy.

## HTTP regression

Authenticated API access, a256-token administrator cap,1583 restored prefix
tokens, a14-request burst (12 admitted,2 rejected with429), and disconnect
cleanup passed. Closing a stream before consuming its first generated token
returned all slots to idle in1.99seconds. This verifies reclamation, not
instantaneous transport-disconnect detection or interruption of an in-flight
ANE call.

## Deployment

The validated binary and v8 asset replaced the per-user service on18083.
It binds0.0.0.0, serves gemma4-e2b-ane-batch, and keeps concurrency4,
context16384, output default1024, administrator output ceiling4096,
prefix cache128MiB, queue8, deadline120seconds and temperature0.
The new prefill budget is1. LAN-address readiness and Chinese translation
passed. The prior binary, model and service configuration remain available
for rollback. The first launchd registration attempt rolled back successfully;
waiting for job removal and old-PID exit before registration allowed the
subsequent update to succeed. This was not a reboot test.
