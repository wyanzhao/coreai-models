# Native KV experiment, November 3, 2026

The direct runtime-layout reader did not pass the performance gate. Production
remains at 44cc503 with the v8 asset, batch 4, q64 prefill and budget 1.
The candidate is retained behind explicit --native-kv; it defaults off.
After timing, the CLI gate changed from experimental default-on to opt-in;
the measured reader algorithm is unchanged and final parity was rerun.
Mutable-state probes are separate artifacts and are not used by llm-server.

All exports, builds, inference and measurements ran on the M6 Mac mini
(16GiB, macOS27.0.1). Host activity was SSH control and source publication only.

## Measured runtime comparison

Each run restarts its server, warms up, and checks compiler activity. Requests
have distinct prefixes and zero reused prefix tokens. Final comparison order
was old/native/native/old. Numbers below average the two runs; all 56 cases
across three reader variants are retained in [native-kv-results.json](native-kv-results.json).

| 4K nominal input | Old TTFT seconds | Native TTFT seconds | Old aggregate E2E tok/s | Native aggregate E2E tok/s |
|---|---:|---:|---:|---:|
| output128, concurrency1 | 1.028 | 1.072 | 34.61 | 34.19 |
| output128, concurrency4 | 3.836 | 4.038 | 61.26 | 60.28 |
| output256, concurrency1 | 0.987 | 1.045 | 40.51 | 40.14 |
| output256, concurrency4 | 3.822 | 4.444 | 82.98 | 79.51 |

Actual server prompt counts are recorded per request, approximately 4160.
E2E includes prefill and completion; it is not decode-only throughput.
For output256, decode-only rate was 47.82 tok/s for both single-request paths;
four-request mean per-request rates were 30.36 old and 30.58 native.
Mixed traffic (one1K/output256 request followed by three8K/output128 requests)
completed the short request in 9.016s old versus 9.272s native.

The first generic native reader increased single4K TTFT to about 1.38s.
Power-of-two address arithmetic reduced it to about 1.22-1.27s.
Contiguous-buffer bypass and inlining reduced the cost further, but still
did not beat the accepted baseline. One final native concurrency4 case had
4.875s TTFT; it remains in the mean. The data do not support a speedup claim.

The comparison isolates dense materialization from borrowed physical-layout
reads. Borrowed reads still map values into CPU prefix snapshots and device
input buffers, and still validate finiteness. They do not establish end-to-end
native attention layout or zero-copy state updates.

A runner output-directory migration initially wrote two new sweep JSON files
over old ones. Original per-run probe logs remained intact; all original
cases were recovered from those logs and new cases relocated from their own
logs. The runner now requires a fresh directory and uses it for every output.

## Correctness and state probes

The three reader revisions each passed 16 bitwise logit comparisons across
ragged prefill lengths 32/65/600/1089, row shrink/expansion 4→2→1→4,
ring wrap and prefix/row reuse. Same-graph numerical equality is required. The final build passed 36 tests
in five selected suites, including the independent full-width reference,
ragged batch4, prefix restoration and ring-wrap checks. Batch8 and the older
prefill ablation are environment-gated and were not exercised in this run.

The descriptor-backed state probe uses nonuniform channel/token values.
Global q64/C512/capacity16384 writes at 0/64/128/16320 passed exact output
and state comparison. Sliding q64/C256/capacity576 writes at 0/512/0/64
also passed. The final ring check additionally wrote 0/64/128/192/256/320/
384/448/512/0/64 with exact output/state at every step. These are
micrographs, not full Gemma4 inference.

For a rank5 fused q1 probe, repeated offset 0 and offsets 0/4/8/60 passed
on ANE; offset 1 returned request-cancelled status 0x12. GPU control at
0/1/2/63 passed. A rank4 B2 independent-offset probe crashed during ANE
specialization. Neither finding is a universal ANE hardware specification.

Both q64 micrographs have 9 ANE and 3 GPU residency entries. The mutable
update source maps to ANE; remaining GPU work and possible bridging/copies
must be measured. Placement counts are not utilization or zero-copy proof.

## Reproduction and next acceptance gate

Use [run_native_comparison.py](run_native_comparison.py) with a fresh
--output-dir, --baseline-binary and a local --api-key-file. O selects the
baseline binary, N selects this binary with --native-kv, D selects this
binary's default dense path. Keep accelerator tests serialized.

Export the B1 state probe with:

```sh
PYTHONPATH=$PWD/python/src USE_OS_COREAI=1 .venv/bin/python ane-server/probe_mutable_kv.py --output ane-server/results/new-state-probe --batch 1 --fused --interleave8 --query 64 --channels 512 --context 16384 --export-only
```
 Run MutableKVProbeTests with
COREAI_STATE_PROBE pointing to model.aimodel and
COREAI_STATE_POSITIONS=0,64,128,16320.
State buffers must be allocated from runtime descriptors, not dense arrays.

Next: multi-layer state layout and attention parity, then aligned q64
prefill integration. Arbitrary prefix restoration and partial tails retain
the validated functional path until alignment and ownership are satisfied.
Decode, cancellation/reuse, prefix isolation and measured whole-model
benefit remain mandatory gates before deployment. See [NATIVE_KV_PLAN.md](NATIVE_KV_PLAN.md).
