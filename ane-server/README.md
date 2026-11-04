# Experimental ANE LLM server

This fork adds a Gemma4 E2B functional-KV backend: independent request state, continuous batching, chunked prefill, bounded immutable prefix snapshots, streaming chat, queue limits and cancellation.

The validated serving graphs support B2/B4/B8 decode, q32/q64 prefill, and a 16384-token total context. The earlier B4/v6 artifact has 23,416 ANE-resident and 12 GPU-resident optimized operations in the compiler metadata. The v8 KV-only artifact reports16,980 ANE and10 GPU operations. This is an ANE-dominant route, not an ANE-only guarantee or a runtime utilization percentage. Constrained JSON-schema decoding and logprob scoring are currently rejected. This is not full OpenAI API compatibility.

## Build and export

Use full Xcode and the repository Python environment. Source weights and generated assets are not included. Initial source revision: google/gemma-4-E2B-it at 4adf15748de9e1a0097b6274cee9c15836c41c37.

    PYTHONPATH=python/src USE_OS_COREAI=1 python models/gemma4/export_batched.py --source /path/to/source --source-revision YOUR_SOURCE_REVISION --output /path/to/new-asset --context 16384 --shapes 1:32,1:64,2:1,4:1 --kv-only-prefill
    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build -c release -j 2 --product llm-server

Export is not numerical validation. The initial16K B1/q1 specialization failed parity; single-request decode and one-token tails use the validated B2 graph with an inert second row.

    swift run -c release llm-server --model /path/to/validated-asset --functional-batch --concurrent-requests 4 --prefix-cache-mib 128 --host 127.0.0.1 --port 18083 --max-tokens 1024 --temperature 0 --max-queue-depth 8 --no-thinking

Endpoints: /v1/chat/completions, /v1/models, /ready, /v1/stats. Cache stats count actual restored KV. Output defaults can be overridden within remaining total context. Batch capacity depends on exported graphs.

## Validation

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter 'BatchSchedulerTests|ReadonlyBatchKVTests|ServerAPITypesTests'

Functional integration tests additionally use COREAI_FUNCTIONAL_ASSET and COREAI_FUNCTIONAL_CASE. COREAI_FUNCTIONAL_LONG=1 enables ring-wrap tests. COREAI_FUNCTIONAL_BATCH4=1 enables four-row compaction and q64 parity; COREAI_FUNCTIONAL_BATCH8=1 enables eight-row compaction for a separately exported B8 asset. Ordinary unit tests do not download models.

Earlier v5/q32 M6 16GiB baseline (not a controlled A/B with the newer backend): decode about46-47 tokens/s for one request,45-46 per request for two. Nominal1K/4K/8K input with256-token outputs: end-to-end aggregate37.3/23.7/16.1 tokens/s for one request,60.9/32.0/19.4 for two. Prefill is serialized in bounded chunks. Single observations are not p95 or hardware-peak measurements.

## Server controls

The experimental branch now discovers B2/B4 decode and q32/q64 prefill capabilities from the exported manifest. Vacant slots are compacted into the smallest supported decode graph. New graphs still require numerical validation.

- --prefill-chunk-size selects the maximum exported query width used for prefill.
- --max-output-tokens is an administrator ceiling, unlike --max-tokens which is only a default. Context room also caps output; reaching the effective cap reports finish_reason=length.
- --request-timeout-seconds bounds batched queue plus inference time;0 disables. Deadlines are checked between prefill chunks/decode steps, not by interrupting an in-flight accelerator call. Non-streaming expiration returns504; streaming expiration emits an error event.
- --api-key-env reads an optional bearer token from a named environment variable. No token is printed or accepted on the command line. Liveness/readiness remain public; API routes and statistics require the token when enabled.
- stream_options.include_usage=true adds usage:null to ordinary SSE chunks and a final choices:[] usage chunk before[DONE]. Failures are explicit SSE error events.
- Prefix caching reserves a quarter of practical budgets for short shared prefixes. Sparse long checkpoints reduce snapshot copy work. The cache remains bounded and in-memory.

Use serve.py with a JSON configuration to launch the Release binary:

    python3 ane-server/serve.py --config ane-server/config.example.json --check
    python3 ane-server/serve.py --config ane-server/config.example.json

The example binds localhost. Set host explicitly for LAN serving. Do not run a second instance on an occupied port. Authentication is optional; configure the named environment variable in the supervising process when enabling it.

benchmark_server.py runs cold-prefix translation requests at multiple concurrency levels, checks SSE usage against emitted token counts, and records TTFT/decode/end-to-end rates separately. Model weights, credentials, raw logs and local benchmark results are excluded from this repository.


## B4 serving measurements

M6 16GB, macOS 27.0.1, Gemma4 E2B shipping palettization, 16K total context,
q64 prefill, cold distinct prefixes. Values below are one run per case, with
256 generated tokens per request. Prompt labels are nominal; actual server
token counts include the chat template (about 64 additional tokens).

| Nominal input | Concurrent requests | Aggregate end-to-end tokens/s | Mean TTFT seconds | Mean decode tokens/s per request |
|---|---:|---:|---:|---:|
| 1K | 1 | 40.5 | 0.98 | 47.8 |
| 1K | 2 | 68.1 | 1.95 | 45.9 |
| 1K | 4 | 84.1 | 3.85 | 30.8 |
| 4K | 1 | 28.8 | 3.56 | 47.8 |
| 4K | 2 | 40.3 | 7.13 | 45.9 |
| 4K | 4 | 45.5 | 14.00 | 30.2 |
| 8K | 1 | 20.6 | 7.10 | 47.7 |
| 8K | 2 | 26.2 | 13.93 | 45.4 |
| 8K | 4 | 27.4 | 27.72 | 26.5 |

Increasing concurrency trades individual latency for throughput. Prefill is
chunked but still serialized across requests, so long prompts limit scaling.
These measurements do not establish p95 latency or full ANE utilization.

## Operational checks and login service

The HTTP checks use a dedicated test instance with an administrator output cap
of 256 tokens and queue depth 8. They verify the cap, prefix reuse, 429
backpressure and slot reclamation after client disconnects:

    python3 ane-server/http_checks.py --key-file /private/path/test-key --output /path/http-checks.json

For a validated local deployment, install a per-user launchd job after ensuring
the selected port is free:

    python3 ane-server/install_service.py --config /path/to/local-config.json

The installer refuses to overwrite an existing plist. It starts the service
now and on user login, and restarts it after exit; it is not a system boot
daemon. Authenticated deployments should use a supervisor that supplies the
secret environment variable; this helper deliberately does not persist keys.
The generated plist names the exact binary and model paths. Reinstall or edit
the supervised command when changing those paths.

The load probe records ANECompilerService CPU deltas and optional server RSS
(--server-pid). Treat compiler_quiet=false cases as contaminated; compiler
inactivity alone does not establish complete system isolation.

## Functional test fixture

ane-server/fixtures/gemma4-token-ids.json supplies a small fixed token seed for
the ragged-row, compaction and q64 tests. With the appropriate exported asset:

    COREAI_FUNCTIONAL_ASSET=/path/to/asset COREAI_FUNCTIONAL_CASE=ane-server/fixtures/gemma4-token-ids.json COREAI_FUNCTIONAL_BATCH4=1 DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test -j 2 --filter 'fourRaggedRowsMatchIndependentDecode|widerPrefillMatchesOriginalChunking'

For B8, set COREAI_FUNCTIONAL_BATCH8=1 and filter
eightRaggedRowsMatchIndependentDecode. Its reference rows run independently
through the smaller graph, and its live-row sequence is 8 → 4 → 2 → 1 → 8.
The separate realModelRaggedRowsAndPrefixRestore test also requires externally
captured logits.out.bin beside its case.json; that reference is not bundled.

Keep deployed models and the binary in a real directory under Workspace, outside Documents/Desktop/Downloads. launchd does not inherit the terminal's protected-folder approval. The installer resolves symlinks and rejects these known protected locations before starting a job.

See [the same-artifact batch4/8 comparison](BATCH_COMPARISON.md). Concurrency4 remains the default. config.batch8.example.json is an opt-in template for the separately validated B8 asset.

## Prefill traffic and scheduling

The opt-in exporter flag --kv-only-prefill emits ABI v2. Its q32/q64 entrypoints
return only new KV tensors; the existing q1 tail produces the final prompt
logits. ABI v1 assets remain supported. Each entrypoint's manifest outputs
are checked against the loaded runtime function.

The functional backend keeps one persistent prefill KV bank per occupied row.
It reuses the bank only when all physical NDArray descriptors match, restores
valid cached prefix positions on binding, and writes only valid new KV tokens
after each chunk. Completion, cancellation and errors release the bank.
This reduces host packing; it does not assert ANE SRAM residency or eliminate
the final q1 tail's full-cache input preparation.

--disable-persistent-prefill retains the old packing path for controlled
comparisons. --prefill-chunks-per-step bounds the total chunks before the next
decode step, with round-robin service across prefilling rows. The functional
server defaults to1;0 visits every prefilling row as before. This is a work
budget, not preemption of an in-flight ANE call. With q64 and budget1, at most
64 valid prefill tokens are processed between decode steps.

The real-model prefillOptimizationAblations test accepts
COREAI_PREFILL_BASELINE alongside COREAI_FUNCTIONAL_ASSET and the token fixture.
It compares the original asset, headless prefill and persistent buffers. The
mixed HTTP probe benchmark_mixed.py introduces three8K requests after a1K
request starts decoding, then records token gaps and completion times.

See [the prefill optimization comparisons](PREFILL_RESULTS.md) for the v8 default, independent ablations, correctness gates and mixed-request latency.

## Native KV experiments

[Measured results and acceptance gates](NATIVE_KV_RESULTS.md) cover direct
runtime-layout reads and explicit mutable-state probes. The --native-kv
option is experimental and defaults off; no stateful graph is deployed.

## q64 stateful prefill

The separate q64 state asset is integrated but remains opt-in. Full-model
correctness passed; the HTTP comparison did not establish a stable speedup,
so production retains v8 without stateful prefill.

- [Integration and reproduction](STATEFUL_PREFILL.md): export, configuration,
  correctness tests and comparison runner.
- [Measured results](STATEFUL_PREFILL_RESULTS.md) and
  [machine-readable metrics](stateful-prefill-results.json).
- [Original integration plan](STATEFUL_PREFILL_PLAN.md).

The combined-asset variant is an unfinished experiment retained locally;
it is not included in the supported export or serving path.
