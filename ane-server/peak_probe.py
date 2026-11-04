"""Compute-bound FP16 lower-bound probe, not a claim of hardware peak TOPS.

Run on the target Mac with USE_OS_COREAI=1. FMA counts as two floating-point
operations. Compilation, torch reference and tensor setup are outside timing.
Final placement and FP32-reference SNR gate every reported throughput.
"""
import argparse, asyncio, collections, copy, json, statistics, time
from pathlib import Path

import numpy as np
import torch
from coreai_torch import TorchConverter
from coreai.runtime import AIModel, NDArray, SpecializationOptions, ComputeUnitKind


async def probe(root, channels, width, depth, repeats):
    torch.manual_seed(17)
    layers = []
    for _ in range(depth):
        conv = torch.nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            conv.weight.normal_(std=channels ** -0.5)
        layers.extend([conv, torch.nn.ReLU()])
    module = torch.nn.Sequential(*layers).eval().half()
    x = torch.randn(1, channels, 1, width).half()
    with torch.no_grad():
        exported = torch.export.export(module, (x,)).run_decompositions(torch.export.default_decompositions())
        reference = copy.deepcopy(module).float()(x.float()).numpy()
    asset = root / ("conv-c{}-s{}-d{}.aimodel".format(channels, width, depth))
    TorchConverter().add_exported_program(exported, input_names=["x"], output_names=["y"]).to_coreai().save_asset(asset)
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine()).with_debug(enabled=True)
    model = await AIModel.load(asset, options)
    tables = json.loads(model._debug_infos)
    placement = collections.Counter()
    for table in tables:
        if table.get("identifier") != "MPSGraph_optimized_op_id":
            continue
        for operation in table.get("operations", []):
            values = [m["value"]["string"]["_0"] for m in operation.get("metadatas", []) if m.get("key") == "residency"]
            if values:
                placement[values[-1]] += 1
    fn = model.load_function(model.function_names[0])
    inputs = [{"x": NDArray(x.numpy())} for _ in range(2)]
    result = await fn(inputs[0])
    actual = result["y"].numpy().astype(np.float32)
    signal = float(np.square(reference.astype(np.float64)).sum())
    noise = float(np.square(actual.astype(np.float64)-reference).sum())
    snr = -10*np.log10(max(noise/max(signal, 1e-30), 1e-30))
    accepted = bool(np.isfinite(actual).all() and snr >= 40 and placement["ANE"] > 0 and placement["GPU"] == 0)
    results = []
    for concurrency in [1, 2]:
        for _ in range(3):
            await asyncio.gather(*(fn(inputs[i]) for i in range(concurrency)))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            outputs = await asyncio.gather(*(fn(inputs[i]) for i in range(concurrency)))
            elapsed = time.perf_counter() - start
            # Consume after await; verify completion/finite output outside timing.
            assert all(np.isfinite(o["y"].numpy()).all() for o in outputs)
            samples.append(elapsed)
        median = statistics.median(samples)
        operations = 2*depth*channels*channels*width*concurrency
        results.append({"concurrency": concurrency, "median_s": median, "samples_s": samples,
                        "effective_fp16_tflops": operations/median/1e12 if accepted else None})
    row = {"channels": channels, "width": width, "depth": depth, "snr_db": float(snr),
           "placement": dict(placement), "accepted": accepted, "results": results}
    (root / (asset.stem + ".json")).write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    return row


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", default="512,1024,2048")
    parser.add_argument("--widths", default="32,128,512,2048")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in map(int, args.channels.split(",")):
        for width in map(int, args.widths.split(",")):
            rows.append(await probe(args.output, c, width, args.depth, args.repeats))
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
