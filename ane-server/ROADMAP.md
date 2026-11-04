# ANE server development acceptance plan

Base: validated Gemma4 E2B functional-KV backend. Preserve the running B2 service until each replacement passes correctness and HTTP lifecycle tests.

1. Publish the validated source on the user fork; keep weights, compiled assets, private machine instructions and raw logs local.
2. Export B2/B4 decode plus q32/q64 prefill. Discover graph shapes from manifest, pad to the smallest validated graph, and reject unsupported geometry. Never execute known-invalid 16K B1/q1.
3. Validate ragged rows, multi-step generation, ring wrap, cache restore, mixed prefill/decode and cancellation before throughput admission. Compare B1/B2/B4, then B8 only when B4 shows room and memory is safe.
4. Improve short-prefix retention and prefill buffer traffic. Add serving features needed for sustained translation: streamed usage, request limits/timeouts, optional authentication, configuration/startup, metrics and documented capability errors. Evaluate constrained generation/scoring separately; do not advertise unimplemented API compatibility.
5. Establish peak methodology: official hardware specifications, compute-bound FP16 matmul shape sweep with final ANE placement and numerical checks, concurrency scaling, then model efficiency. Count FMA as two FLOPs. Public marketing multipliers are not FP16 TFLOPS. No claim of full ANE utilization without a measured plateau and placement evidence.

All export, build, inference and benchmark work runs on the mini. Device experiments are serialized to avoid measuring concurrent compiler load. GitHub/browser access is administrative control only.
