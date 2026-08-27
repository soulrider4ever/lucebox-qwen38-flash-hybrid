#!/usr/bin/env python3
import argparse, hashlib, json, statistics, threading, time, urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="http://127.0.0.1:8088")
parser.add_argument("--repetitions", type=int, default=2)
parser.add_argument("--profile", default="unknown")
parser.add_argument("--print-long-prompt", action="store_true")
parser.add_argument("--long-records", type=int, default=53)
args = parser.parse_args()

BASE = args.base
MODEL = "Qwen3.8-Flash-Next-IQ4_XS"
PCI = "0000:c5:00.0"

def fdinfo(path):
    d = {}
    try:
        for line in Path(path).read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1); d[k.strip()] = v.strip()
    except OSError:
        pass
    return d

def find_pid():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try: cmd = p.joinpath("cmdline").read_bytes().decode(errors="ignore")
        except OSError: continue
        if "llama-server" in cmd and "--port\x008088" in cmd: return int(p.name)
    raise RuntimeError("8088 server pid not found")

def memory_bytes(pid):
    clients = {}
    for path in Path(f"/proc/{pid}/fdinfo").glob("*"):
        d = fdinfo(path)
        if d.get("drm-driver") != "amdgpu" or d.get("drm-pdev") != PCI: continue
        cid = d.get("drm-client-id", path.name)
        def n(key):
            try: return int((d.get(key) or "0").split()[0])
            except (ValueError, IndexError): return 0
        val = (n("drm-total-vram") + n("drm-total-gtt")) * 1024
        clients[cid] = max(clients.get(cid, 0), val)
    return sum(clients.values())

def run(pid, prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role":"user", "content":prompt}],
            "max_tokens":max_tokens, "temperature":0, "stream":True,
            "cache_prompt":False,
            "chat_template_kwargs":{"enable_thinking":False,"preserve_thinking":False}}
    req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    stop = threading.Event(); peak = [0]; samples = [0]
    def monitor():
        while not stop.is_set():
            peak[0] = max(peak[0], memory_bytes(pid)); samples[0] += 1; stop.wait(.02)
        peak[0] = max(peak[0], memory_bytes(pid))
    mon = threading.Thread(target=monitor, daemon=True); mon.start()
    output=[]; usage={}; timings={}; first=None; start=time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line=raw.decode("utf-8","replace").strip()
                if not line.startswith("data:"): continue
                val=line[5:].strip()
                if val == "[DONE]": continue
                try: obj=json.loads(val)
                except json.JSONDecodeError: continue
                if obj.get("usage"): usage=obj["usage"]
                if obj.get("timings"): timings=obj["timings"]
                for choice in obj.get("choices",[]):
                    content=(choice.get("delta") or {}).get("content")
                    if content:
                        if first is None: first=time.perf_counter()-start
                        output.append(content)
    finally:
        wall=time.perf_counter()-start; stop.set(); mon.join(2)
    prompt_n=int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    output_n=int(usage.get("completion_tokens") or timings.get("predicted_n") or len(output))
    prompt_ms=float(timings.get("prompt_ms") or timings.get("prefill_ms") or 0)
    decode_ms=float(timings.get("predicted_ms") or timings.get("decode_ms") or 0)
    if not all((prompt_n, output_n, prompt_ms, decode_ms, first is not None)):
        raise RuntimeError(json.dumps({"usage":usage,"timings":timings,"first":first}))
    text="".join(output)
    return {"prompt_tokens":prompt_n,"output_tokens":output_n,"ttft_ms":first*1000,
            "tokSOut":output_n/(decode_ms/1000),"tokSPrefill":prompt_n/(prompt_ms/1000),
            "tokSTotal":(prompt_n+output_n)/wall,"wall_ms":wall*1000,
            "peakVramGb":peak[0]/1e9,"samples":samples[0],
            "sha256":hashlib.sha256(text.encode()).hexdigest(),"timings":timings,
            "draft_n":usage.get("draft_n"),"draft_n_accepted":usage.get("draft_n_accepted")}

def synthetic_prompt(records=53):
    rows=[]
    for i in range(1, records + 1):
        rows.append(f"Record {i}: a local inference server streams transformer weights from unified memory, tracks recurrent state checkpoint {i % 32}, and verifies draft group {(i % 7)+1}. The observation code is R{i:03d}-{(i*7919)%104729:06d}.\n")
    return "".join(rows)+"\nIn one concise paragraph, summarize the system described above and state the highest record number and its observation code."

if args.print_long_prompt:
    print(synthetic_prompt(args.long_records), end="")
    raise SystemExit(0)

tasks=[("count_to_30","Count from 1 to 30, comma separated, nothing else.",250),
       (f"synthetic_{args.long_records}_records_to_128",synthetic_prompt(args.long_records),128)]
pid=find_pid(); report={"pid":pid,"pci":PCI,"model":MODEL,"profile":args.profile,"tasks":[]}
for name,prompt,n in tasks:
    warm=run(pid,prompt,n); measured=[run(pid,prompt,n) for _ in range(args.repetitions)]
    med=lambda k: statistics.median(x[k] for x in measured)
    row={"name":name,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"warmup":warm,"runs":measured,
         "median":{"promptTokens":int(statistics.median(x["prompt_tokens"] for x in measured)),"outputTokens":int(statistics.median(x["output_tokens"] for x in measured)),
         "tokSOut":med("tokSOut"),"tokSPrefill":med("tokSPrefill"),"tokSTotal":med("tokSTotal"),"ttftMs":med("ttft_ms"),"peakVramGb":max(x["peakVramGb"] for x in measured),"outputHashes":sorted(set(x["sha256"] for x in measured))}}
    report["tasks"].append(row); print(json.dumps({"task":name,"median":row["median"]}),flush=True)
report["localmaxxing_candidate"]=report["tasks"][-1]["median"]
print("FINAL "+json.dumps(report),flush=True)
