"""Measure decode stalls while three long translation prompts enter the server."""
import argparse, concurrent.futures, json, subprocess, threading, time, urllib.request, uuid
from pathlib import Path
from transformers import AutoTokenizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="http://127.0.0.1:18084")
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--api-key-file",type=Path)
    a=ap.parse_args()
    headers={"Content-Type":"application/json"}
    if a.api_key_file: headers["Authorization"]="Bearer "+a.api_key_file.read_text().strip()
    def get(path):
        with urllib.request.urlopen(urllib.request.Request(a.base+path,headers=headers)) as f:return json.load(f)
    model=get("/v1/models")["data"][0]["id"]
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    unit="The research team carefully recorded the temperature and compared the observations with earlier results."
    corpus=tok.encode("\n".join(f"{i}. {unit}" for i in range(1000)),add_special_tokens=False)
    started=threading.Event()
    def send(index):
        prompt=1024 if index==0 else 8192
        body={"model":model,"messages":[{"role":"system","content":uuid.uuid4().hex+". Translate every numbered paragraph into Chinese. Return only translations."},{"role":"user","content":tok.decode(corpus[:prompt-80])}],"temperature":0,"max_tokens":256 if index==0 else 128,"stream":True,"raw":True,"stream_options":{"include_usage":True}}
        begin=time.monotonic(); stamps=[];usage=None;done=False
        req=urllib.request.Request(a.base+"/v1/chat/completions",headers=headers,data=json.dumps(body).encode())
        with urllib.request.urlopen(req,timeout=180) as f:
            for line in f:
                if not line.startswith(b"data: "):continue
                raw=line[6:].strip()
                if raw==b"[DONE]":done=True;break
                event=json.loads(raw)
                if "error" in event:raise RuntimeError(event["error"])
                if event.get("usage"):usage=event["usage"]
                if event["choices"] and event["choices"][0]["delta"].get("content") is not None:
                    stamps.append(time.monotonic())
                    if index==0:started.set()
        assert done and usage and len(stamps)==usage["completion_tokens"]
        gaps=sorted(b-a for a,b in zip(stamps,stamps[1:]))
        return {"index":index,"ttft_s":stamps[0]-begin,"total_s":time.monotonic()-begin,"usage":usage,"max_gap_s":max(gaps,default=0),"p95_gap_s":gaps[int((len(gaps)-1)*.95)] if gaps else 0,"token_times":stamps}
    def compiler_times():
        result={}
        for line in subprocess.check_output(["ps","-axo","pid,time,comm"],text=True).splitlines():
            if "ANECompilerService" not in line:continue
            pid,value,_=line.split(None,2)
            seconds=0.0
            for part in value.split(":"):seconds=seconds*60+float(part)
            result[pid]=seconds
        return result
    before=get("/v1/stats")
    compiler_before=compiler_times()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        short=pool.submit(send,0)
        if not started.wait(60):raise TimeoutError("short request did not begin decoding")
        longs=[pool.submit(send,i) for i in range(1,4)]
        results=[short.result()]+[f.result() for f in longs]
    compiler_after=compiler_times()
    delta=sum(v-compiler_before.get(k,0) for k,v in compiler_after.items())
    record={"before":before,"after":get("/v1/stats"),"requests":results,
        "compiler_cpu_seconds":delta,"compiler_quiet":compiler_before.keys()==compiler_after.keys() and 0<=delta<0.5}
    a.output.write_text(json.dumps(record,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!="token_times"} for r in results]),flush=True)
if __name__=="__main__":main()
