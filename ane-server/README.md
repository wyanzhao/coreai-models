# Experimental ANE LLM server

This fork adds a Gemma4 E2B functional-KV backend: independent request state, continuous batching, chunked prefill, bounded immutable prefix snapshots, streaming chat, queue limits and cancellation.

The validated baseline uses B2 and a 16384-token total context. It is ANE-dominant with some GPU fallback. Constrained JSON-schema decoding and logprob scoring are currently rejected. This is not full OpenAI API compatibility.

## Build and export

Use full Xcode and the repository Python environment. Source weights and generated assets are not included. Initial source revision: google/gemma-4-E2B-it at 4adf15748de9e1a0097b6274cee9c15836c41c37.

    python models/gemma4/export_batched.py --source /path/to/source --output /path/to/new-asset --context 16384
    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build -c release --product llm-server

Export is not numerical validation. The initial16K B1/q1 specialization failed parity; single-request decode and one-token tails use the validated B2 graph with an inert second row.

    swift run -c release llm-server --model /path/to/validated-asset --functional-batch --concurrent-requests 2 --prefix-cache-mib 128 --host 127.0.0.1 --port 18083 --max-tokens 1024 --temperature 0 --max-queue-depth 8 --no-thinking

Endpoints: /v1/chat/completions, /v1/models, /ready, /v1/stats. Cache stats count actual restored KV. Output defaults can be overridden within remaining total context. Batch capacity depends on exported graphs.

## Validation

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter 'BatchSchedulerTests|ReadonlyBatchKVTests|ServerAPITypesTests'

Functional integration tests additionally use COREAI_FUNCTIONAL_ASSET and COREAI_FUNCTIONAL_CASE. COREAI_FUNCTIONAL_LONG=1 enables ring-wrap tests. Ordinary unit tests do not download models.

Initial M6 16GiB measurements: decode about46-47 tokens/s for one request,45-46 per request for two. Nominal1K/4K/8K input with256-token outputs: end-to-end aggregate37.3/23.7/16.1 tokens/s for one request,60.9/32.0/19.4 for two. Prefill is serialized in bounded chunks. Single observations are not p95 or hardware-peak measurements.
