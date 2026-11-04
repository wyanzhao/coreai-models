"""Archive compiled placement metadata; counts are not runtime utilization."""
import argparse, asyncio, gzip, hashlib, json
from collections import Counter
from pathlib import Path
from coreai.runtime import AIModel, SpecializationOptions, ComputeUnitKind

async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--asset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    options=SpecializationOptions.from_preferred_compute_unit_kind(
        ComputeUnitKind.neural_engine()).with_debug(enabled=True)
    model=await AIModel.load(args.asset.resolve()/"model.aimodel",options)
    raw=model._debug_infos
    digest=hashlib.sha256(raw).hexdigest()
    archive=args.output.with_suffix(".debug.json.gz")
    with gzip.open(archive,"wb",compresslevel=1) as stream:stream.write(raw)
    tables=json.loads(raw)
    summary=[]
    for table in tables:
        counts=Counter()
        for op in table.get("operations",[]):
            residencies=[m["value"]["string"]["_0"] for m in op.get("metadatas",[])
                         if m.get("key")=="residency"]
            if residencies:counts[residencies[-1]]+=1
        summary.append({"table":{k:v for k,v in table.items() if k!="operations"},
                        "last_residency_counts":dict(counts)})
    checksum=hashlib.sha256()
    with gzip.open(archive,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):checksum.update(block)
    assert checksum.hexdigest()==digest
    result={"asset":str(args.asset.resolve()),"raw_sha256":digest,"archive_verified":True,"tables":summary}
    args.output.with_suffix(".summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":asyncio.run(main())
