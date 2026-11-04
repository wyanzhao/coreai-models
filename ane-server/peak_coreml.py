"""Independent ANE compute probe using Core ML CPU_AND_NE (GPU excluded).

This is an achieved FP16 throughput lower bound on a different runtime route,
not the CoreAI server's speed and not an official hardware TOPS specification.
"""
import argparse, collections, concurrent.futures, json, statistics, time
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def probe(root, c, width, depth, repeats):
    torch.manual_seed(17)
    layers=[]
    for _ in range(depth):
        conv=torch.nn.Conv2d(c,c,1,bias=False)
        with torch.no_grad():conv.weight.normal_(std=c**-0.5)
        layers.extend([conv,torch.nn.ReLU()])
    module=torch.nn.Sequential(*layers).eval().half().float()
    x=torch.randn(1,c,1,width).half()
    with torch.no_grad():
        reference=module(x.float()).numpy()
        traced=torch.jit.trace(module,x.float(),check_trace=False)
    model=ct.convert(traced,inputs=[ct.TensorType(name="x",shape=tuple(x.shape),dtype=np.float16)],
        outputs=[ct.TensorType(name="y",dtype=np.float16)],convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE)
    asset=root/f"conv-c{c}-s{width}-d{depth}.mlpackage"
    model.save(str(asset))
    plan=MLComputePlan.load_from_path(model.get_compiled_model_path(),compute_units=ct.ComputeUnit.CPU_AND_NE)
    placements=[];conv_devices=[]
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
        device=type(usage.preferred_compute_device).__name__ if usage else None
        placements.append({"operator":op.operator_name,"preferred_device":device})
        if op.operator_name.rsplit(".",1)[-1]=="conv":conv_devices.append(device)
    actual=model.predict({"x":x.numpy()})["y"].astype(np.float32)
    signal=float(np.square(reference.astype(np.float64)).sum())
    noise=float(np.square(actual.astype(np.float64)-reference).sum())
    snr=-10*np.log10(max(noise/max(signal,1e-30),1e-30))
    accepted=bool(len(conv_devices)==depth and all(d=="MLNeuralEngineComputeDevice" for d in conv_devices)
        and np.isfinite(actual).all() and snr>=40)
    second=ct.models.MLModel(str(asset),compute_units=ct.ComputeUnit.CPU_AND_NE)
    models=[model,second];inputs=[{"x":x.numpy().copy()} for _ in models]
    timings=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        def run(i):return models[i].predict(inputs[i])
        for concurrency in [1,2]:
            for _ in range(3):list(pool.map(run,range(concurrency)))
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter()
                outputs=list(pool.map(run,range(concurrency)))
                samples.append(time.perf_counter()-start)
                assert all(np.isfinite(o["y"]).all() for o in outputs)
            median=statistics.median(samples)
            timings.append({"concurrency":concurrency,"median_s":median,"samples_s":samples,
                "effective_fp16_tflops":2*depth*c*c*width*concurrency/median/1e12 if accepted else None})
    row={"runtime":"CoreML CPU_AND_NE","channels":c,"width":width,"depth":depth,
        "snr_db":float(snr),"accepted":accepted,"placements":placements,"results":timings}
    (root/(asset.stem+".json")).write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--channels",default="1024,2048")
    ap.add_argument("--widths",default="256,1024,4096")
    ap.add_argument("--depth",type=int,default=8)
    ap.add_argument("--repeats",type=int,default=15)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in map(int,args.channels.split(",")):
        for w in map(int,args.widths.split(",")):
            rows.append(probe(args.output,c,w,args.depth,args.repeats))
            (args.output/"summary.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__":main()
