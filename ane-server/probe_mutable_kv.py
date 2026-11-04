"""Explicit-state persistence/independent-offset probe; not a full-model result."""
import asyncio,json,argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from coreai_torch import TorchConverter
from coreai.authoring.types import HardwareConstraints,AllocationType
from coreai.runtime import AIModel,NDArray,SpecializationOptions,ComputeUnitKind
from coreai_models.primitives._ops import mutable_slice_update,mutable_cache_update_and_fetch
from coreai_models.export.mlir_ops import register_custom_torch_lowering,remove_functionalization

class Probe(torch.nn.Module):
    def __init__(self,batch,channels,capacity):
        super().__init__();self.batch=batch;self.channels=channels;self.capacity=capacity;self.query=1
    def forward(self,updates,positions,cache):
        if getattr(self,"static",False):positions=torch.zeros_like(positions)
        for row in range(self.batch):
            zero=torch.zeros(1,dtype=torch.int32,device=positions.device)
            row_t=zero+row
            begin=torch.cat((row_t,zero,zero,positions[row:row+1]))
            end=torch.cat((row_t+1,zero+self.channels,zero+1,positions[row:row+1]+self.query))
            mutable_slice_update(cache,updates[row:row+1],begin,end)
        return cache.clone()

class FusedProbe(Probe):
    def forward(self,updates,positions,cache):
        if getattr(self,"static",False):positions=torch.zeros_like(positions)
        rows=[]
        for row in range(self.batch):
            zero=torch.zeros(1,dtype=torch.int32,device=positions.device)
            begin=torch.cat((zero+row,zero,zero,zero,positions[row:row+1]))
            end=torch.cat((zero+row+1,zero+1,zero+self.channels,zero+1,positions[row:row+1]+self.query))
            rows.append(mutable_cache_update_and_fetch(cache,updates[row:row+1],begin,end,row,-1,None))
        return torch.cat(rows,dim=0)

async def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--batch",type=int,choices=[1,2],default=2);ap.add_argument("--static",action="store_true");ap.add_argument("--interleave8",action="store_true");ap.add_argument("--export-only",action="store_true");ap.add_argument("--fused",action="store_true");ap.add_argument("--query",type=int,default=1);ap.add_argument("--channels",type=int,default=32);ap.add_argument("--context",type=int,default=64);a=ap.parse_args()
    if a.query < 1 or a.channels < 1 or a.context < max(64,a.query):
        ap.error("positive query/channels and context >= max(64, query) required")
    a.output.mkdir(parents=True,exist_ok=False)
    b,c,n=a.batch,a.channels,a.context
    model=(FusedProbe if a.fused else Probe)(b,c,n).eval()
    model.static=a.static
    model.query=a.query
    samples=(torch.zeros(b,c,1,a.query,dtype=torch.float16),torch.arange(b,dtype=torch.int32)*4,torch.zeros((b,1,c,1,n) if a.fused else (b,c,1,n),dtype=torch.float16))
    ep=torch.export.export(model,samples).run_decompositions()
    remove_functionalization(ep)
    conv=TorchConverter();register_custom_torch_lowering(conv)
    conv.add_exported_program(ep,input_names=["updates","positions"],state_names=["cache"],output_names=["observed"])
    program=conv.to_coreai()
    if a.interleave8:
        program.set_hardware_constraints("main",{"cache":HardwareConstraints(AllocationType.IOSurface,interleave=[1,1,8,1,1] if a.fused else [1,8,1,1],alignments=[1,1,1,1,8*n,1] if a.fused else [1,1,1,8*n,1])})
    program.save_asset(a.output/"model.aimodel")
    if a.export_only:return
    if a.query != 1 or a.fused:raise ValueError("Use the Swift descriptor-backed probe for this geometry")
    options=SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine()).with_debug(enabled=True)
    runtime=await AIModel.load(a.output/"model.aimodel",options)
    fn=runtime.load_function("main")
    cache=NDArray(np.zeros((b,c,1,n),dtype=np.float16));state={"cache":cache}
    reference=np.zeros((b,c,1,n),dtype=np.float16);checks=[]
    for step,positions in enumerate(([0,3],[1,7],[2,8],[63,62])):
        values=np.full((b,c,1,1),step+1,dtype=np.float16);values[-1]*=2
        positions=[0]*b if a.static else positions[:b]
        outputs=await fn({"updates":NDArray(values),"positions":NDArray(np.array(positions,dtype=np.int32))},state=state)
        for row,pos in enumerate(positions):reference[row,:,:,pos:pos+1]=values[row]
        actual=outputs["observed"].numpy()
        exact=np.array_equal(reference,actual)
        checks.append({"step":step,"exact":exact,"max_error":float(np.max(np.abs(actual-reference)))})
        print(json.dumps(checks[-1]),flush=True)
    tables=json.loads(runtime._debug_infos);placement=[]
    for t in tables:
        counts=Counter()
        for op in t.get("operations",[]):
            rs=[m["value"]["string"]["_0"] for m in op.get("metadatas",[]) if m.get("key")=="residency"]
            if rs:counts[rs[-1]]+=1
        placement.append({"identifier":t.get("identifier"),"counts":dict(counts)})
    result={"checks":checks,"placement":placement,"state_shape":list(cache.shape)}
    result["passed"]=all(x["exact"] for x in checks)
    (a.output/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=="__main__":asyncio.run(main())
