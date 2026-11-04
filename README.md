# Core AI Models — ANE translation server

This fork of [apple/coreai-models](https://github.com/apple/coreai-models) adds a
Gemma 4 E2B text-generation server for a Mac mini, including an OpenAI-style
Chat Completions endpoint for Immersive Translate.

The default branch is **experiment/gemma4-ane-batching**. Its accepted v8 route
uses KV-only intermediate prefill, persistent per-request KV buffers and bounded
prefill scheduling. It is **ANE-dominant, not ANE-only**: some compiled operations
run on the GPU. The implementation and model exports remain experimental.

- [Features and limits](#features-and-limits)
- [Export and build](#export-and-build)
- [Deployment configuration](#deployment-configuration)
- [Immersive Translate](#immersive-translate)
- [Measured performance](#measured-performance)
- [Validation and benchmarking](#validation-and-benchmarking)
- [Upstream components](#upstream-components)

## Features and limits

| Area | Supported on the functional-batch route |
|---|---|
| Validated serving model | Gemma 4 E2B Instruct, text input/output, mixed 4-bit/8-bit palettization |
| Context | 16,384 tokens **total**, including template, input and output |
| Decode batching | Independent request state; continuous batching; compact active rows into B2/B4 graphs |
| Prefill | q32/q64 chunks; one globally scheduled chunk between decode steps by default |
| KV memory | Persistent prefill buffers; descriptor-aware physical layout; bounded immutable prefix snapshots |
| Prefix reuse | In-memory token-prefix reuse across requests; default budget 128 MiB |
| API | Streaming and non-streaming POST /v1/chat/completions; GET /v1/models, /ready and /v1/stats |
| Lifecycle | EOS/stop handling, bounded queue, cancellation cleanup, request deadline and output cap |
| Operations | Configurable bind address/model ID; optional bearer authentication via environment variable; JSON launcher and per-user launchd installer |

Decode capacity 4 is the interactive default. B8 graphs passed separate
correctness checks, but require a separate export and trade higher throughput
for longer response latency; changing the request count alone cannot create
missing model entrypoints. Prefill chunks are still executed serially across
requests. This is not parallel execution of four independent ANE model calls.

**Not advertised on this route:** full OpenAI API compatibility, tool calling,
constrained JSON-schema decoding, logprob scoring, vision/audio serving, or
validated serving of other model families. The upstream model catalog has a
broader scope than this custom backend.

The experimental native-KV reader and q64 graph-state updates are not enabled
on this branch. The separate
[experiment/stateful-q64 branch](https://github.com/wyanzhao/coreai-models/tree/experiment/stateful-q64)
contains their follow-up evidence: q64 state passed full-model correctness but
did not establish a stable speedup. Persistent host-side prefill buffers in v8
are a different, accepted optimization.

## Export and build

### Requirements

- Apple silicon Mac, macOS 27.0+ and full Xcode 27.0+ with its license accepted.
- Measured setup: Mac mini M6, 16 GiB, macOS 27.0.1 (26A434), Xcode SDK 27.2,
  Python 3.12, torch 2.9.0, coreai-core 1.0.0b3 and coreai-torch 0.4.3.
- uv and access to the gated
  [google/gemma-4-E2B-it checkpoint](https://huggingface.co/google/gemma-4-E2B-it).
  Accept its license and authenticate with Hugging Face on the export machine.
- Space for source weights, mapped FP16 weights, exported assets and compiler
  caches. Weights and generated model bundles are not committed.

Keep the repository and assets under Workspace when using launchd; its process
does not inherit Terminal's access to Documents/Desktop/Downloads.

~~~sh
mkdir -p "$HOME/Workspace"
cd "$HOME/Workspace"
git clone --branch experiment/gemma4-ane-batching https://github.com/wyanzhao/coreai-models.git
cd coreai-models
uv sync --python 3.12
~~~

Use Python 3.12 explicitly with uv run too: the upstream .python-version may
otherwise select a different interpreter.

### Download the pinned source

The measured revision is 4adf15748de9e1a0097b6274cee9c15836c41c37.
After authenticating with `hf auth login`:

~~~sh
uv run --python 3.12 hf download google/gemma-4-E2B-it \
  --revision 4adf15748de9e1a0097b6274cee9c15836c41c37 \
  --local-dir models-source/gemma4-e2b \
  --include model.safetensors \
  --include config.json \
  --include tokenizer.json \
  --include tokenizer_config.json \
  --include generation_config.json \
  --include chat_template.jinja
~~~

The custom exporter expects a single model.safetensors file and all five
JSON/template resources above. Preserve the canonical chat_template.jinja;
a generic template is not a validated replacement.

### Export the v8 functional-batch bundle

Use **models/gemma4/export_batched.py** for this server, rather than the
upstream static-engine export.py. The output directory must be new.

~~~sh
PYTHONPATH="$PWD/python/src" USE_OS_COREAI=1 \
  uv run --python 3.12 python models/gemma4/export_batched.py \
  --source models-source/gemma4-e2b \
  --source-revision 4adf15748de9e1a0097b6274cee9c15836c41c37 \
  --output ane-assets/gemma4-e2b-b4-c16384-v8-kvonly \
  --context 16384 \
  --shapes 1:32,1:64,2:1,4:1 \
  --kv-only-prefill
~~~

Default compression uses the shipping mixed 4-bit/8-bit palettization recipe,
with two workers. Keep the default automatic KV layout. The bundle contains
model.aimodel, metadata.json, batch-manifest.json, tokenizer/template resources,
embeddings and PLE sidecars; deploy the **whole directory**.

Intermediate q32/q64 calls return KV updates only; the q1 tail produces logits.
The earlier 16K B1/q1 graph failed parity, so the validated single-row path
uses B2/q1 with an inert second row. Export success is not a correctness pass:
validate a newly exported bundle before replacing a service. Runtime
specialization occurs on the target Mac and can make first startup slow.

### Build

~~~sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build -c release -j 2 --product llm-server
~~~

The JSON launcher expects .build/out/Products/Release/llm-server. Run the
binary with --help for the full CLI; the configuration fields below cover the
accepted translation setup.

## Deployment configuration

Create a local configuration in the ignored results directory:

~~~sh
mkdir -p ane-server/results
cp ane-server/config.example.json ane-server/results/translation-config.json
~~~

**Change model to an absolute path** in this copy: its directory differs from
the example's directory.
Relative model paths are always resolved relative to the configuration file.

A complete LAN configuration is:

~~~json
{
  "model": "/absolute/path/to/coreai-models/ane-assets/gemma4-e2b-b4-c16384-v8-kvonly",
  "functional_batch": true,
  "host": "0.0.0.0",
  "port": 18083,
  "server_model_name": "gemma4-e2b-ane-batch",
  "concurrent_requests": 4,
  "prefill_chunk_size": 64,
  "prefill_chunks_per_step": 1,
  "prefix_cache_mib": 128,
  "max_queue_depth": 8,
  "max_tokens": 1024,
  "max_output_tokens": 4096,
  "request_timeout_seconds": 120,
  "temperature": 0,
  "no_thinking": true
}
~~~

| Setting | Meaning |
|---|---|
| host | 127.0.0.1 for local clients; 0.0.0.0 for all interfaces, or the Mac's LAN address |
| concurrent_requests | Up to 4 active requests with this export; additional requests wait in the bounded queue |
| prefill_chunks_per_step | Global chunk budget before decode; 1 favors interactive progress; 0 visits every prefilling row |
| max_tokens | Default output budget when the client does not provide one |
| max_output_tokens | Administrator output ceiling; client requests are also limited by remaining context |
| request_timeout_seconds | Queue plus inference deadline; checked between accelerator calls, not mid-call |
| prefix_cache_mib | Memory budget for prefix snapshots; not an on-disk cache |

16K is not 16K input plus 4K output. Reserve space for the template and output;
the example permits at most 4096 output tokens, subject to remaining context.

~~~sh
.venv/bin/python ane-server/serve.py \
  --config ane-server/results/translation-config.json --check

.venv/bin/python ane-server/serve.py \
  --config ane-server/results/translation-config.json
~~~

For optional bearer authentication, add `"api_key_env": "ANE_SERVER_API_KEY"`
and set that variable in the launching process. The value is not a JSON field
or CLI argument. Without it, the service is unauthenticated. Use the LAN bind
only on a network where that access is intended; this server does not provide
TLS. /ready stays public; configured authentication protects API routes and stats.

### Check from the MacBook

Use the **mini's** address, not localhost on the MacBook. The measured setup
uses 192.168.2.3; substitute your server address.

~~~sh
curl -fsS http://192.168.2.3:18083/ready
curl -fsS http://192.168.2.3:18083/v1/models
curl -fsS http://192.168.2.3:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-e2b-ane-batch","messages":[{"role":"user","content":"Translate to Chinese. Output only the translation: Good morning."}],"max_tokens":64,"temperature":0}'
~~~

When authentication is enabled, add
`-H "Authorization: Bearer $ANE_SERVER_API_KEY"` to API requests.
Port 18083 is this fork's accepted service; older experiments on 18081/18082
use other backends/configurations.

### Optional launchd service

After a foreground smoke test, stop that instance before installing a service:

~~~sh
.venv/bin/python ane-server/install_service.py \
  --config ane-server/results/translation-config.json \
  --label local.coreai.ane-server
~~~

The installer creates a per-user LaunchAgent with RunAtLoad/KeepAlive. It refuses
to overwrite an existing plist or install an authenticated configuration;
for authentication, use a supervisor that supplies the environment securely.
Existing deployments need an explicit backup/update/rollback procedure.
Logs are written to ane-server/results/service.stdout.log and service.stderr.log.
Startup was tested; reboot behavior was not tested. The server Mac must remain
awake and reachable while clients use it.

## Immersive Translate

In the browser extension's translation-service settings, choose an OpenAI
compatible/custom AI service. The
[official custom-address instructions](https://immersivetranslate.com/docs/services/openai/#customizing-openais-api-address)
describe the entry point; labels can vary by extension version.

| Field | Value for the measured LAN setup |
|---|---|
| Custom API URL | http://192.168.2.3:18083/v1/chat/completions |
| Custom model | gemma4-e2b-ane-batch |
| API key | The configured bearer token; if server auth is disabled and the UI requires a value, use a non-secret placeholder such as local |
| Temperature | 0 |

The custom API URL is the full completion endpoint. A client that specifically
asks for a **base URL** instead should use http://192.168.2.3:18083/v1.

Suggested starting settings below are tuning guidance, **not an extension-wide
quality benchmark**:

| Extension option | Starting point |
|---|---|
| Maximum requests per second | 2; increase toward 4 only if latency/queue statistics stay acceptable |
| Maximum paragraphs per batch | 4; try 8 for short paragraphs |
| Subtitle batch size | 4 for quicker first results |
| AI batch translation | Enhanced mode, preserving the extension's required output format |
| AI smart context | Off initially to avoid an extra analysis request |
| Rich text | Enable if links/styles are needed; disable temporarily when diagnosing malformed output |
| AI expert | General; retain the extension's default batch prompts initially |

Requests per second is an arrival-rate limit, **not batch size or concurrency**.
A server with 4 active slots and 8 waiting slots will return 429 when its queue
fills. Raising the extension's rate does not make the ANE execute more work at
once. Shorter batches can improve responsiveness; overly small batches add
request/template overhead.

If translation stalls, check /ready, /v1/stats, server logs and the browser
console together. A page spinner alone does not establish slow inference.
For 429 reduce arrival rate; for 504 reduce batch size/work or inspect queue
delay; for finish_reason=length reduce text per batch or request more output
within the server/context limits. If curl succeeds but the extension fails,
check its custom URL/model, site or local-network permissions and page state.

## Measured performance

Measurements below were made on the Mac mini M6 / 16 GiB / macOS 27.0.1 setup
above, with the pinned E2B checkpoint, q64 prefill and 16K context.
**These are measured workload rates, not peak ANE TOPS or utilization.**
The v8 compiler metadata places 16,980 operations on ANE and 10 on GPU;
operation counts are not execution-time fractions.

### Accepted v8 versus the earlier v6 baseline

Four concurrent requests, 256 output tokens each. Throughput is aggregate
**output tokens / entire HTTP batch wall time**, including prefill.

| Nominal input | v6 end-to-end tok/s | v8 end-to-end tok/s | Gain | v6 mean TTFT | v8 mean TTFT |
|---|---:|---:|---:|---:|---:|
| 1K | 84.15 | 105.63 | 25.5% | 3.808 s | 1.206 s |
| 4K | 46.03 | 83.23 | 80.8% | 14.004 s | 3.815 s |
| 8K | 28.46 | 64.72 | 127.4% | 27.610 s | 7.332 s |

The restart-isolated A,D,D,A,B,C,C,B campaign had 64 compiler-quiet,
cold-prefix observations. Table entries average two observations; these are
not population p95 estimates. Inputs are nominal; actual server token counts
include template overhead. Single-request decode remains about **48 tok/s**.

At 4K / concurrency4 / TG256, removing intermediate logits improves throughput
from 46.03 to 67.70 tok/s; adding persistent prefill buffers reaches 85.74 tok/s.
The global budget of one chunk reduces that to 83.23 tok/s but improves mixed
workloads: when three long requests arrive, the short request completes in
9.04 s rather than 12.37 s, and its within-stream p95 token gap drops from
62 ms to 36 ms.

See [the full ablation report](ane-server/PREFILL_RESULTS.md),
[raw summary metrics](ane-server/prefill-results.json) and
[earlier B4/B8 comparison](ane-server/BATCH_COMPARISON.md). The B8 comparison
used an older v7 asset; it is not a v8 B4/B8 comparison.

### Effective prefill rate

A later cold-prefix single-request measurement of the same accepted v8 route
estimated rate as **actual input tokens / HTTP time to first token**:

| Nominal input | Effective rate | Mean TTFT |
|---|---:|---:|
| 1K | 3,394 input tok/s | 0.322 s |
| 4K | 4,129 input tok/s | 1.008 s |
| 8K | 4,295 input tok/s | 1.923 s |

Each row uses six requests across TG128/256, with roughly 64 extra input tokens
from templating. This includes HTTP, scheduling and first-token work; it is
**not isolated model prefill TPS**. It belongs to a separate campaign from the
table above. [Source campaign and limitations](https://github.com/wyanzhao/coreai-models/blob/84e36c111ff972d0cc47313144430322e47a783e/ane-server/STATEFUL_PREFILL_RESULTS.md).

## Validation and benchmarking

~~~sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test -c release -j 2 \
  --filter 'BatchSchedulerTests|ReadonlyBatchKVTests|ServerAPITypesTests'
~~~

Full-model tests additionally require COREAI_FUNCTIONAL_ASSET and
COREAI_FUNCTIONAL_CASE, pointing to the exported bundle and
ane-server/fixtures/gemma4-token-ids.json. COREAI_FUNCTIONAL_LONG=1 enables
ring-wrap tests; COREAI_FUNCTIONAL_BATCH4=1 enables four-row checks.
A unit-only run is not a full-model validation. See
[the validation record](ane-server/VALIDATION.md).

For a new HTTP matrix, use a dedicated test instance/port, warm it first, then
collect nominal PP1024/4096/8192 × TG128/256 with three repeats at concurrency1/4:

~~~sh
.venv/bin/python ane-server/benchmark_server.py \
  --base http://127.0.0.1:18084 \
  --tokenizer ane-assets/gemma4-e2b-b4-c16384-v8-kvonly \
  --output ane-server/results/my-warmup.json \
  --prompts 1024,4096,8192 --outputs 128,256 --concurrency 1,4

.venv/bin/python ane-server/benchmark_server.py \
  --base http://127.0.0.1:18084 \
  --tokenizer ane-assets/gemma4-e2b-b4-c16384-v8-kvonly \
  --output ane-server/results/my-timed-run.json \
  --prompts 1024,4096,8192 --outputs 128,256 --concurrency 1,4 \
  --repeats 3 --server-pid SERVER_PID
~~~

Replace SERVER_PID with the test server's PID and use fresh output filenames;
these scripts can overwrite existing JSON. Add --api-key-file for authenticated
test instances. Check compiler_quiet, reused_prefix_tokens, authoritative usage
and finish reasons. Randomized system prefixes prevent repeated warmup prompts
from making cold PP look artificially fast. Do not benchmark while exports,
compiler audits or another load test are competing for the accelerator.

benchmark_mixed.py measures interference from arriving long prompts.
run_prefill_comparison.py reproduces the named v6/v8 ablation sequence, expects
both assets, and uses the fixed results/prefill-comparison directory: archive
existing results before running it. No new model export or performance run is
implied by the documentation update.

## Upstream components

This repository retains Apple's model export recipes, Python primitives,
Swift runtime utilities and agent skills. Other families use their own
[model-specific recipes](models/); their support does not imply compatibility
with this fork's functional-batch server.

| Directory | Contents |
|---|---|
| models/ | Model catalog and export recipes |
| python/ | PyTorch authoring primitives and export helpers |
| swift/ | Core AI runtime utilities, CLI tools and tests |
| ane-server/ | Server configuration, deployment, benchmarks and evidence |
| skills/ | Core AI agent skills |

List the upstream model catalog with:

~~~sh
uv run --python 3.12 coreai.model.registry --list-models
~~~

For app integration, exported models require Core AI on macOS/iOS 27+ and
model-specific resources; see each family's README for narrower requirements.
The upstream model catalog and skills remain available alongside this server.

## Agent Skills

This repo includes a plugin with skills to enable coding agents to use Core AI like an expert.

### Available skills

| Skill | Description |
| --- | --- |
| `working‑with‑coreai` | End-to-end workflow for deploying PyTorch models on Apple silicon, covering export with `coreai-torch` and running with the Core AI runtime. |
| `model‑authoring` | Empirical rules for authoring PyTorch models for on-device execution on Apple platforms, covering BC1S layout, op compatibility, KV cache patterns, precision rules, MoE, and common issues. |
| `model‑compression‑exploration` | Systematically explore weight compression configurations (quantization and palettization) for a PyTorch model using `coreai-opt`. |

### Install

Installation differs depending on your coding agent of choice.

#### Claude Code

Register the marketplace:

```
/plugin marketplace add git@github.com:apple/coreai-models.git
```

Alternatively, register the marketplace from a local git checkout:

```
/plugin marketplace add /path/to/coreai-models
```

Install the plugin:

```
/plugin install coreai-skills@coreai-models
```

#### Codex CLI

Register the marketplace:

```
codex plugin marketplace add https://github.com/apple/coreai-models
```

Alternatively, register the marketplace from a local git checkout:

```
codex plugin marketplace add /path/to/coreai-models
```

Launch Codex in your workspace:

```
codex
```

Install the plugin through the interactive browser: once the Codex session is
active in your terminal, open the plugin manager by typing `/plugins`, locate the
`coreai-models` marketplace tab (use your arrow keys or the built-in search),
select `coreai-skills`, and choose Install.

#### Gemini CLI

Install the extension from a local directory:

```
gemini extensions install /path/to/coreai-models/skills
```

Once installed, the skills activate automatically based on your task context,
or you can invoke them explicitly.

## Contributing

We welcome bug fixes, new models, documentation and test improvements within a
defined scope. Read
[CONTRIBUTING.md](https://github.com/apple/coreai-models/blob/main/CONTRIBUTING.md)
before you open a pull request.
Report bugs and request models through the
[issue templates](https://github.com/apple/coreai-models/issues/new/choose).

## Support

- [GitHub Issues](https://github.com/wyanzhao/coreai-models/issues) — Feedback, bug reports, and feature requests

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
