# Validation record

Target: Mac mini, Apple M6, 16GiB, macOS 27.0.1 (26A434).
Model: google/gemma-4-E2B-it revision
4adf15748de9e1a0097b6274cee9c15836c41c37.
All exports, builds and inference were performed on the serving Mac.

## Accepted correctness checks

- 28 scheduler, KV-cache and graph-plan tests plus 17 API serialization tests.
- B4 ragged prompts at 32, 65, 577 and 640 tokens; live rows transition
  4 → 2 → 1 → 4 → 2 → 4. All 17 row-step comparisons were bitwise equal
  to independent smaller-graph references.
- B8 ragged prompts at 32, 65, 577, 640, 33, 129, 513 and 600 tokens;
  live rows transition 8 → 4 → 2 → 1 → 8. All 23 row-step comparisons
  were bitwise equal to independent smaller-graph references.
- B8 asset also passed independent full-width prefill-reference, ragged B2,
  multi-step decode and prefix-restore checks.
- q64 versus q32 chunking: four comparisons at 46.57–61.52 dB SNR,
  all with the same top token. Different chunk boundaries are not required
  to be bitwise identical; the established gate is 40 dB plus matching top1.
- The earlier 16K B1/q1 graph failed correctness and remains excluded.
  Single-request decode uses B2 with an inert second row.

## HTTP behavior

On the authenticated B4 test instance:
- absent/wrong bearer tokens returned 401; the configured token succeeded;
- a requested 1000-token output was capped to 256 with finish_reason=length;
- the repeated 1584-token prompt restored 1583 actual KV prefix tokens;
- a 14-request burst admitted 12 and rejected 2 with 429;
- disconnecting streaming clients released all active and queued requests;
- streamed usage counts matched raw generated-token event counts in all 18
  load-test cases (1K/4K/8K nominal input, 128/256 output, concurrency1/2/4).

With a deliberately tiny deadline, non-streaming returned 504 and streaming
returned an explicit request_timeout SSE event; both released capacity.

## Limits of the evidence

These are focused correctness and operational tests, not a translation-quality
benchmark or exhaustive coverage of every prompt and context position.
Timing tables are single observations per case, not p95 claims.
The backend still serializes bounded prefill chunks across requests.
Constrained JSON-schema decoding and logprob scoring are explicitly unsupported
on this route; tool-calling compatibility is not advertised.
Compiled placement metadata is not a runtime FLOP or utilization percentage.

B4/v6 final optimized-operation placement metadata: 23,416 ANE and 12 GPU. These are operation counts across compiled entrypoints, not weighted execution-time fractions.

B8/v7 final optimized-operation placement metadata: 29,248 ANE and 15 GPU.

The B8 HTTP instance passed the same cap/prefix checks. Its18-request burst
admitted16 and returned429 to2; all slots and queue entries were reclaimed
after disconnects. The same-artifact concurrency4/8 sweep completed all12
cases (1K/4K/8K,128/256 output) with compiler CPU deltas at most0.01seconds
per case. The B4 deployment was reached over LAN and returned the expected
Chinese translation. launchd startup was tested after moving model assets
outside Documents; login/reboot behavior is configured, not reboot-tested.
