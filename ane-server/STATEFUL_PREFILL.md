# q64 stateful prefill

## Status and scope

This optional route updates persistent KV state in complete, naturally aligned
q64 prefill calls. Prefix restoration preserves baseline chunk boundaries;
unaligned continuations, tails and final logits use the functional route.
It does not change decode batching or enable the separate native-KV reader.

[Correctness and performance results](STATEFUL_PREFILL_RESULTS.md) record the
acceptance decision: correct, but no stable performance gain. Keep it disabled
for serving. The example configuration targets the separate experimental port
18084; it is not the production configuration.

## Export

Run from the repository root with the existing Python environment and full
Xcode on the target Mac. Use a fresh output directory and the same pinned
source revision, compression, context and cache geometry as the base asset.

    PYTHONPATH=python/src USE_OS_COREAI=1 .venv/bin/python models/gemma4/export_batched.py --source /path/to/gemma4-e2b --source-revision 4adf15748de9e1a0097b6274cee9c15836c41c37 --output /path/to/new-q64-state-asset --context 16384 --shapes 1:64 --kv-only-prefill --stateful-prefill

The runtime checks compatibility against the base functional asset. Export or
compiler placement alone does not establish numerical correctness.

## Build and validate

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build -c release -j 2 --product llm-server

    COREAI_FUNCTIONAL_ASSET=/path/to/v8-functional-asset COREAI_STATEFUL_ASSET=/path/to/q64-state-asset COREAI_FUNCTIONAL_CASE="$PWD/ane-server/fixtures/gemma4-token-ids.json" DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test -c release -j 2 --filter StatefulPrefillTests

The integration test returns without model execution when required environment
variables are absent. Confirm full-model case output and state engagement
counts in the log before claiming a model validation pass.

Do not run placement audits concurrently with correctness or timed runs.
CoreAI cache errors are recorded in the results report; do not purge an active
serving asset to recover an experimental loader.

## Launch and compare

Copy config.stateful.example.json and adjust both model paths. Relative paths
are resolved from the configuration file's directory.

    .venv/bin/python ane-server/serve.py --config /path/to/stateful-config.json --check
    .venv/bin/python ane-server/serve.py --config /path/to/stateful-config.json

Stop the temporary instance before running the comparison driver, which owns
port 18084 and starts/stops its own server processes:

    .venv/bin/python ane-server/run_stateful_comparison.py --api-key-file /path/to/private-test-key --baseline-binary /path/to/accepted/llm-server --state-asset /path/to/q64-state-asset --output-dir /path/to/new-comparison-directory --sequence OSSO --prompts 1024,4096,8192 --repeats 3

O uses the supplied baseline binary; D uses the current Release binary with
state disabled; S uses that binary with state enabled. Both functional routes
use ane-assets/gemma4-e2b-b4-c16384-v8-kvonly. The output directory must not
already exist. Each variant warms all cases before timed TG128/256,
concurrency1/4 requests and a mixed-traffic probe.

Check compiler activity, prefix reuse, actual token counts and state counters.
Prompt budgets are nominal; requests are randomized independently between
variants. HTTP TTFT includes scheduling and is not pure PP execution time.
The completed campaign used OS; the default OSSO is a proposed stronger
repeat order, not a claim about the published data.

## Retained unfinished work

The Mac mini worktree retains
ane-server/results/combined-state-and-paired-benchmark-wip.patch and the
associated export/load logs. That patch combines model entrypoints and adds
deterministic request seeds, but its full-model loader validation is blocked.
It is deliberately excluded from the validated source tree. Raw artifacts,
weights, credentials and compiled models remain local.
