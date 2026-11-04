"""Exercise real HTTP limits/cache; source stays on the serving machine."""
import argparse,json,urllib.request,urllib.error,concurrent.futures,time
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument("--base",default="http://127.0.0.1:18084")
ap.add_argument("--key-file",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
ap.add_argument("--cap",type=int,default=256)
ap.add_argument("--queue-depth",type=int,default=8)
args=ap.parse_args()
headers={"Content-Type":"application/json","Authorization":"Bearer "+args.key_file.read_text().strip()}
def call(path,body=None):
    request=urllib.request.Request(args.base+path,headers=headers,data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(request,timeout=150) as response:return json.load(response)
model=call("/v1/models")["data"][0]["id"]
unit="The research team measured the temperature every morning. They compared the results with previous observations and carefully recorded the differences. The report explains how the new procedure improves reliability and reduces unnecessary work."
body={"model":model,"messages":[{"role":"system","content":"Translate all numbered paragraphs into simplified Chinese. Preserve numbering. Return only translations."},{"role":"user","content":"\n".join(f"{i}. {unit}" for i in range(1,40))}],"max_tokens":1000,"temperature":0,"raw":True}
start=time.monotonic()
a=call("/v1/chat/completions",body)
cold_seconds=time.monotonic()-start
before=call("/v1/stats")
start=time.monotonic()
b=call("/v1/chat/completions",body)
warm_seconds=time.monotonic()-start
after=call("/v1/stats")
result={"cold_seconds":cold_seconds,"warm_seconds":warm_seconds,"first":a,"warm":b,"before":before,"after":after,"ready":call("/ready")}
args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
assert a["usage"]["completion_tokens"]<=args.cap
assert b["usage"]["completion_tokens"]<=args.cap
assert after["reused_prefix_tokens"]>before["reused_prefix_tokens"]
assert a["usage"]["completion_tokens"]==args.cap and a["choices"][0]["finish_reason"]=="length"
print(json.dumps({"cap":a["usage"],"finish":a["choices"][0]["finish_reason"],"warm_reused_tokens":after["reused_prefix_tokens"]-before["reused_prefix_tokens"]}))


# Hold response bodies until the burst has been admitted, then disconnect after
# the first generated token. This checks backpressure and capacity reclamation.
burst_size=call("/ready")["batching"]["concurrent_requests"]+args.queue_depth+2
barrier=__import__("threading").Barrier(burst_size)
def cancelled_request(index):
    request_body=dict(body,stream=True,max_tokens=256)
    request_body["messages"]=[dict(m) for m in body["messages"]]
    request_body["messages"][0]["content"]=f"Request {index}. "+request_body["messages"][0]["content"]
    request=urllib.request.Request(args.base+"/v1/chat/completions",headers=headers,data=json.dumps(request_body).encode())
    try:
        response=urllib.request.urlopen(request,timeout=150)
    except urllib.error.HTTPError as error:
        barrier.wait(timeout=45)
        return error.code
    barrier.wait(timeout=45)
    with response:
        for line in response:
            if line.startswith(b"data: ") and line[6:].strip()!=b"[DONE]":
                event=json.loads(line[6:])
                if event.get("choices") and event["choices"][0].get("delta",{}).get("content") is not None:
                    break
    return 200
with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as pool:
    statuses=list(pool.map(cancelled_request,range(burst_size)))
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    ready=call("/ready")
    if ready["batching"]["active"]==0 and ready["batching"]["queued"]==0:
        break
    time.sleep(.25)
result.update(burst_statuses=statuses,after_cancel_ready=ready)
args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
assert statuses.count(429)>=1,statuses
assert ready["batching"]["active"]==0 and ready["batching"]["queued"]==0,ready
print(json.dumps({"burst":statuses,"after_cancel_ready":ready}))
