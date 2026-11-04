"""Serial, restart-isolated prefill ablations on the serving machine."""
import argparse, hashlib, json, os, subprocess, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"ane-server/results/prefill-comparison"
ASSETS=ROOT/"ane-assets"
PYTHON=str(ROOT/".venv/bin/python")
BASE="http://127.0.0.1:18084"
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--api-key-file",type=Path,required=True)
    args=parser.parse_args()
    key=args.api_key_file.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    evidence={"base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_diff_sha256":hashlib.sha256(subprocess.check_output(["git","diff"])).hexdigest(),
        "binary_sha256":hashlib.sha256((ROOT/".build/out/Products/Release/llm-server").read_bytes()).hexdigest()}
    (OUT/"identity.json").write_text(json.dumps(evidence,indent=2))
    variants={"A":("v6",False,0),"B":("v8-kvonly",False,0),"C":("v8-kvonly",True,0),"D":("v8-kvonly",True,1)}
    for number,name in enumerate(["A","D","D","A","B","C","C","B"]):
        suffix,persistent,budget=variants[name]
        model=(ASSETS/("gemma4-e2b-b4-c16384-"+suffix)).resolve()
        stem=f"{number}-{name}"
        args=[str(ROOT/".build/out/Products/Release/llm-server"),"--model",str(model),"--functional-batch","--concurrent-requests","4","--prefix-cache-mib","128","--prefill-chunk-size","64","--prefill-chunks-per-step",str(budget),"--host","127.0.0.1","--port","18084","--server-model-name","gemma4-e2b-test","--max-tokens","1024","--max-output-tokens","256","--max-queue-depth","8","--request-timeout-seconds","120","--temperature","0","--no-thinking","--api-key-env","ANE_SERVER_TEST_KEY"]
        if not persistent:args.append("--disable-persistent-prefill")
        env=os.environ.copy();env["ANE_SERVER_TEST_KEY"]=key.read_text().strip()
        headers={"Authorization":"Bearer "+env["ANE_SERVER_TEST_KEY"]}
        print("START",stem,flush=True)
        with (OUT/(stem+"-server.log")).open("wb") as log:
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                deadline=time.monotonic()+600
                while True:
                    if process.poll() is not None:raise RuntimeError(f"server exited {process.returncode}")
                    try:
                        with urllib.request.urlopen(urllib.request.Request(BASE+"/v1/models",headers=headers),timeout=2):break
                    except (OSError,TimeoutError):
                        if time.monotonic()>deadline:raise TimeoutError("server startup")
                        time.sleep(1)
                common=["--base",BASE,"--tokenizer",str(model),"--api-key-file",str(key)]
                with (OUT/(stem+"-probe.log")).open("wb") as probe:
                    subprocess.run([PYTHON,"ane-server/benchmark_server.py",*common,"--output",str(OUT/(stem+"-warm.json")),"--prompts","1024","--outputs","128","--concurrency","4"],check=True,stdout=probe,stderr=subprocess.STDOUT)
                    for attempt in range(2):
                        result=OUT/(stem+f"-sweep-{attempt}.json")
                        subprocess.run([PYTHON,"ane-server/benchmark_server.py",*common,"--output",str(result),"--prompts","1024,4096,8192" if name in "AD" else "4096","--outputs","128,256","--concurrency","1,4","--server-pid",str(process.pid)],check=True,stdout=probe,stderr=subprocess.STDOUT)
                        if all(r["compiler_quiet"] for r in json.loads(result.read_text())):break
                    else:raise RuntimeError("Repeated compiler contamination")
                    subprocess.run([PYTHON,"ane-server/benchmark_mixed.py",*common,"--output",str(OUT/(stem+"-mixed.json"))],check=True,stdout=probe,stderr=subprocess.STDOUT)
                print("DONE",stem,flush=True)
            finally:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    (OUT/"complete.json").write_text(json.dumps({"complete":True,"sequence":"ADDABCCB"}))
if __name__=="__main__":main()
