# ANE performance: evidence and measurement contract

Apple lists a Dual 16-core Neural Engine for M6, meaning two 16-core engines, not two scalar cores. The Mac mini specifications list153GB/s unified-memory bandwidth for the16GB configuration and170GB/s for24/32GB configurations.

Sources:
- https://www.apple.com/mac-mini/specs/
- https://images.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/

The announcement says up to2x Neural Engine performance versus the preceding generation. Neither of those pages supplies a numerical FP16 TFLOPS peak or a precision-qualified TOPS specification. Do not multiply an older marketed TOPS number by two and present it as a measured or official M6 FP16 peak. GPU Neural Accelerators are separate from ANE.

## Measurement

peak_probe.py exports stacked FP16 BC1S1x1 convolutions with ReLU between layers, checks against FP32 computation of the rounded weights, inspects final placement metadata, warms the graph, and measures one and two concurrent calls.

Reported useful matrix FLOPs are2 * depth * channels^2 * width * concurrency. FMA counts as two. ReLU work is excluded. Compilation/reference/setup are outside timing. Accepted results need finite output, at least40dB SNR and no final GPU placement.

The result is achieved effective FP16 compute throughput for these graphs, a lower bound on the hardware's theoretical peak. Python/runtime overhead, intermediate traffic and compiler scheduling remain included. A shape/concurrency plateau is evidence of this workload's saturation, not proof every ANE execution unit is occupied.

## Model serving

Server throughput additionally includes KV traffic, sampling, large-vocabulary logits, tokenization and prefill scheduling. Batching can amortize weight reads without raising single-request speed. Report TTFT, decode throughput, aggregate end-to-end throughput and memory separately. Compare capacity1/2/4 before exploring8, and retain only numerically accepted graphs.

## M6 measurements (2026-11-03)

The CoreAI probe exposed GPU residency metadata, so it is **not** accepted as
an ANE-only peak measurement. The independent Core ML probe uses CPU_AND_NE
to exclude GPU and checks the compute plan for all eight convolution layers
plus numerical parity against FP32 computation of FP16-rounded inputs/weights.

For channels=2048, width=4096, depth=8:
- one model instance: 16.29 effective FP16 TFLOP/s;
- two independent model instances: 22.26 effective FP16 TFLOP/s;
- FP32-reference SNR: 51.65 dB;
- all eight convolution operations have ANE as their preferred compute device.

These are medians of 15 warm end-to-end predictions, including Python,
runtime dispatch and data transfer. FMA counts as two FLOPs. They establish
a measured workload throughput, not an official theoretical peak, a runtime
occupancy percentage, or proof of equal utilization of both Neural Engines.
Other resident services were idle, not uninstalled. CPU fallback remains
permitted by CPU_AND_NE; the compute plan is the placement evidence retained.

The 4096-channel, width=1024 case measured 14.72 / 15.92 TFLOP/s for one/two
instances, with 48.86 dB SNR. Larger dimensions are not automatically faster.
The sweep has not established a universal saturation plateau.

Run the independent route with an isolated coremltools 9 environment:

    python ane-server/peak_coreml.py --output /path/to/results

The source probe and placement records make the operation count and acceptance
criteria inspectable. Do not derive LLM token throughput by dividing this
number by parameter count: the serving graph also pays KV, logits and host
scheduling costs and uses a different runtime route.

The 4096-channel, width4096 point did not finish within a 10-minute sweep budget. It is excluded. Cancelling the client did not immediately quiesce the system compiler; subsequent performance runs must wait for that work to drain.
