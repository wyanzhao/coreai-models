"""Serial restart-isolated native KV ablations on the serving machine."""
import argparse, hashlib, json, os, subprocess, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"ane-server/results/native-comparison"
ASSETS=ROOT/"ane-assets"
PYTHON=str(ROOT/".venv/bin/python")
BASE="http://127.0.0.1:18084"
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--api-key-file",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--sequence",default="ONDDNO")
    parser.add_argument("--baseline-binary",type=Path,required=True)
    args=parser.parse_args()
    if not args.sequence or set(args.sequence)-set("OND"):parser.error("sequence must contain only O, N, D")
    out=args.output_dir.resolve()
    key=args.api_key_file.resolve()
    out.mkdir(parents=True,exist_ok=False)
    evidence={"base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_diff_sha256":hashlib.sha256(subprocess.check_output(["git","diff"])).hexdigest(),
        "binary_sha256":hashlib.sha256((ROOT/".build/out/Products/Release/llm-server").read_bytes()).hexdigest()}
    (out/"identity.json").write_text(json.dumps(evidence,indent=2))
    for number,name in enumerate(args.sequence):
        model=(ASSETS/"gemma4-e2b-b4-c16384-v8-kvonly").resolve()
        binary=args.baseline_binary.resolve() if name=="O" else ROOT/".build/out/Products/Release/llm-server"
        (out/(str(number)+"-"+name+"-binary.sha256")).write_text(hashlib.sha256(binary.read_bytes()).hexdigest())
        stem=f"{number}-{name}"
        command=[str(binary),"--model",str(model),"--functional-batch","--concurrent-requests","4","--prefix-cache-mib","128","--prefill-chunk-size","64","--prefill-chunks-per-step","1","--host","127.0.0.1","--port","18084","--server-model-name","gemma4-e2b-test","--max-tokens","1024","--max-output-tokens","256","--max-queue-depth","8","--request-timeout-seconds","120","--temperature","0","--no-thinking","--api-key-env","ANE_SERVER_TEST_KEY"]
        if name=="N":command.append("--native-kv")
        env=os.environ.copy();env["ANE_SERVER_TEST_KEY"]=key.read_text().strip()
        headers={"Authorization":"Bearer "+env["ANE_SERVER_TEST_KEY"]}
        print("START",stem,flush=True)
        with (out/(stem+"-server.log")).open("wb") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
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
                with (out/(stem+"-probe.log")).open("wb") as probe:
                    subprocess.run([PYTHON,"ane-server/benchmark_server.py",*common,"--output",str(out/(stem+"-warm.json")),"--prompts","1024","--outputs","128","--concurrency","4"],check=True,stdout=probe,stderr=subprocess.STDOUT)
                    for attempt in range(2):
                        result=out/(stem+f"-sweep-{attempt}.json")
                        subprocess.run([PYTHON,"ane-server/benchmark_server.py",*common,"--output",str(result),"--prompts","4096","--outputs","128,256","--concurrency","1,4","--server-pid",str(process.pid)],check=True,stdout=probe,stderr=subprocess.STDOUT)
                        if all(r["compiler_quiet"] for r in json.loads(result.read_text())):break
                    else:raise RuntimeError("Repeated compiler contamination")
                    subprocess.run([PYTHON,"ane-server/benchmark_mixed.py",*common,"--output",str(out/(stem+"-mixed.json"))],check=True,stdout=probe,stderr=subprocess.STDOUT)
                print("DONE",stem,flush=True)
            finally:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    (out/"complete.json").write_text(json.dumps({"complete":True,"sequence":args.sequence}))
if __name__=="__main__":main()
