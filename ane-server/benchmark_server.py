"""HTTP translation load probe; run on the serving Mac for reproducibility."""
import argparse, concurrent.futures, json, os, time, urllib.request, uuid, subprocess
from pathlib import Path
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18083")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--prompts", default="1024,4096,8192")
    ap.add_argument("--outputs", default="128,256")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--server-pid", type=int)
    args = ap.parse_args()
    headers = {"Content-Type": "application/json"}
    if args.api_key_file:
        headers["Authorization"] = "Bearer " + args.api_key_file.read_text().strip()
    def get(path):
        with urllib.request.urlopen(urllib.request.Request(args.base + path, headers=headers), timeout=20) as f:
            return json.load(f)
    model = get("/v1/models")["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    unit = "The research team measured the temperature every morning. They compared the results with previous observations and carefully recorded the differences. The report explains how the new procedure improves reliability and reduces unnecessary work."
    corpus = tokenizer.encode("\n".join(f"{i}. {unit}" for i in range(1,500)), add_special_tokens=False)

    def request_body(prompt, output):
        system = "Translation batch " + uuid.uuid4().hex + ". Translate all numbered paragraphs into simplified Chinese. Preserve numbering. Return only translations."
        overhead = len(tokenizer.apply_chat_template([{"role":"system","content":system},{"role":"user","content":""}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False))
        return {"model":model, "messages":[{"role":"system","content":system},
            {"role":"user","content":tokenizer.decode(corpus[:max(1,prompt-overhead)])}],
            "max_tokens":output, "temperature":0, "stream":True, "raw":True,
            "stream_options":{"include_usage":True}}

    def send(body):
        start = time.monotonic(); first = None; usage = None; events = 0; reason = None; done = False; rid = None
        request = urllib.request.Request(args.base+"/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data: "): continue
                raw = line[6:].strip()
                if raw == b"[DONE]": done = True; break
                event = json.loads(raw)
                if "error" in event: raise RuntimeError(event["error"])
                rid = event["id"]
                if event.get("usage") is not None: usage = event["usage"]
                if not event["choices"]: continue
                choice = event["choices"][0]
                if choice.get("finish_reason"): reason = choice["finish_reason"]
                if choice["delta"].get("content") is not None:
                    if first is None: first = time.monotonic()
                    events += 1
        end = time.monotonic()
        assert done and usage is not None and first is not None
        assert events == usage["completion_tokens"]
        return {"id":rid,"ttft_s":first-start,"total_s":end-start,
            "decode_tps":(events-1)/(end-first) if events>1 else None,
            "usage":usage,"finish_reason":reason}

    def compiler_times():
        output = subprocess.check_output(["ps","-axo","pid,time,comm"], text=True)
        result = {}
        for line in output.splitlines():
            if "ANECompilerService" not in line: continue
            pid, value, _ = line.split(None, 2)
            seconds = 0.0
            for part in value.split(":"): seconds = seconds * 60 + float(part)
            result[pid] = seconds
        return result

    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for prompt in map(int,args.prompts.split(",")):
        for output in map(int,args.outputs.split(",")):
            for concurrency in map(int,args.concurrency.split(",")):
                for repeat in range(args.repeats):
                    bodies = [request_body(prompt,output) for _ in range(concurrency)]
                    before = get("/v1/stats"); compiler_before=compiler_times(); start=time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        results=list(pool.map(send,bodies))
                    elapsed=time.monotonic()-start; after=get("/v1/stats")
                    compiler_after=compiler_times()
                    compiler_delta=sum(value-compiler_before.get(pid,0) for pid,value in compiler_after.items())
                    rss = None
                    if args.server_pid:
                        rss = int(subprocess.check_output(["ps","-p",str(args.server_pid),"-o","rss="],text=True).strip())/1024
                    row={"nominal_prompt":prompt,"requested_output":output,"concurrency":concurrency,"repeat":repeat,
                        "wall_s":elapsed,"aggregate_end_to_end_tps":sum(r["usage"]["completion_tokens"] for r in results)/elapsed,
                        "reused_prefix_tokens":after["reused_prefix_tokens"]-before["reused_prefix_tokens"],
                        "compiler_cpu_seconds":compiler_delta,"compiler_quiet":compiler_before.keys()==compiler_after.keys() and 0<=compiler_delta<0.5,
                        "server_rss_mib_after":rss,"results":results}
                    rows.append(row);args.output.write_text(json.dumps(rows,indent=2))
                    print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()
