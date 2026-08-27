# Qwen3.8 Flash-Next on Lucebox

This document describes the experimental Qwen3.8-Flash-Next work in this
repository and the fastest validated Lucebox profile.

> **Status:** the loader, ownership boundary, and selective R9700 execution
> path are byte- and placement-checked on Lucebox. The ROCm CPU-MoE and
> selective-R9700 profiles are benchmark-only: semantic canaries found that
> both can emit an endless stream of `/` tokens. The production-safe OpenWebUI
> profile is Vulkan on the 8060S with graph reuse, speculative decoding, and
> default reasoning disabled.

## What this fork adds

The Qwen3.8-Flash-Next model uses the Qwen4Exp architecture. Its recurrent/GDN
state, hyper-connections, PLE history, and QSA index cache must remain on one
coherent graph owner. The new Qwen4Exp placement seam therefore limits future
heterogeneous execution to routed MoE expert tensors:

- keeps recurrent state, hyper-connection state, PLE history, and QSA cache on
  the primary owner;
- keeps the shared expert on the primary owner;
- classifies only routed `ffn_*_exps` tensors for hot/cold placement;
- reuses Lucebox's critical-path placement allocator without pretending that
  whole-layer splitting is safe for this architecture.

See [`qwen4exp-hybrid-design.md`](qwen4exp-hybrid-design.md) for the design
and [`server/src/common/qwen4exp_hybrid_plan.cpp`](../server/src/common/qwen4exp_hybrid_plan.cpp)
for the implementation.

## Hardware tested

The reference machine is a Lucebox system with:

- AMD Ryzen AI MAX+ 395 / Radeon 8060S unified-memory GPU (`gfx1151`);
- Radeon AI PRO R9700 (`gfx1201`) attached through OCuLink;
- ROCm 7.x and a 128 GB unified-memory system;
- Ubuntu 26.04 in the reference run.

The validated Qwen3.8 Flash-Next winner uses **only the 8060S**. The R9700 is
hidden from that process with `HIP_VISIBLE_DEVICES=1` because ROCm enumerates
the visible 8060S as `ROCm0` after masking.

## Build

Clone the repository and build the Lucebox server target for the 8060S:

```bash
cmake -S server -B server/build-hip-gfx1151 \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
  -DDFLASH27B_ENABLE_MIXED_CUDA_HIP=OFF \
  -DLLAMA_BUILD_WEBUI=OFF

cmake --build server/build-hip-gfx1151 --target dflash_server -j$(nproc)
```

This builds the Lucebox fork and its placement/test infrastructure. Runtime
generation uses the Qwen4Exp-enabled llama.cpp branch referenced below. That
runtime already contains the native HC/GDN/QSA/PLE graph and hybrid caches;
the fork supplies the split-GGUF inventory, ownership validation, bounded
materialization canaries, and reproducible selective-expert launch contract.

Run the focused fork test after building:

```bash
ctest --test-dir server/build-hip-gfx1151 --output-on-failure \
  -R test_qwen4exp_hybrid_plan
```

For the isolated R9700 loader canaries, build the two dedicated executables
for both Lucebox GPU architectures and pass all split shards in order:

```bash
cmake -S server -B server/build-materializer -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1151;gfx1201'

cmake --build server/build-materializer --target \
  test_qwen4exp_expert_materializer test_qwen4exp_core_loader -j"$(nproc)"

server/build-materializer/test_qwen4exp_expert_materializer \
  /models/qwen38-flash-next/UD-IQ4_XS/*-0000{1,2,3}-of-00003.gguf

server/build-materializer/test_qwen4exp_core_loader \
  /models/qwen38-flash-next/UD-IQ4_XS/*-0000{1,2,3}-of-00003.gguf
```

These executables assume unmasked Lucebox ROCm ordering where device 0 is the
R9700. Verify the printed device identity before trusting the result. They are
bounded load/copy tests; they do not generate tokens.

## Obtain the model

Do not commit model weights. Download the Unsloth GGUF separately:

```bash
huggingface-cli download \
  unsloth/Qwen3.8-Flash-Next-GGUF \
  --include 'UD-IQ4_XS/*' \
  --local-dir /models/qwen38-flash-next
```

The server accepts the first shard as the model path and discovers the
remaining shards in the same directory.

## OpenWebUI-safe profile

Use [`scripts/run-qwen4exp-openwebui-safe.sh`](../scripts/run-qwen4exp-openwebui-safe.sh).
On the reference host:

```bash
LLAMA_SERVER=/path/to/build-vulkan/bin/llama-server \
MODEL=/models/qwen38-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
scripts/run-qwen4exp-openwebui-safe.sh
```

The launcher validates that `Vulkan1` is the Radeon 8060S and then uses:

- all model layers on `Vulkan1`;
- `LLAMA_GRAPH_REUSE_DISABLE=1`;
- `--spec-type none`;
- `--reasoning off`.

Semantic acceptance on 2026-08-27 included a normal greeting for `hi`, an
exact factual response, multi-turn recall, OpenAI-compatible streaming with a
terminal `[DONE]`, and compatibility with requests that still sent the older
ROCm alias. Port 8088 remains unauthenticated and must stay LAN-only.

## Fastest benchmark-only profile

On the reference Lucebox, the fastest measured command was:

```bash
HIP_VISIBLE_DEVICES=1 \
  /path/to/llama-server \
  -m /models/qwen38-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --device ROCm0 \
  --n-gpu-layers 99 \
  --n-cpu-moe 64 \
  --ctx-size 32768 \
  --threads 32 \
  --threads-batch 32 \
  --ubatch-size 1024 \
  --host 0.0.0.0 \
  --port 8088 \
  --jinja \
  --flash-attn on \
  --no-ui \
  --load-mode none \
  --spec-type ngram-mod \
  --alias Qwen3.8-Flash-Next-IQ4_XS-ROCm-CPU-MoE64
```

This profile is **not suitable for chat or correctness evaluation**. Direct
OpenAI-compatible semantic probes returned only `/` tokens with both default
thinking and thinking disabled. The failure persisted after disabling
`ngram-mod` and after setting `LLAMA_GRAPH_REUSE_DISABLE=1`, so neither
speculation nor the local graph-reuse patch is the root cause.

The exact `--device` value is intentionally paired with
`HIP_VISIBLE_DEVICES=1`; without that mask, Lucebox's ROCm device numbering
puts the R9700 first. Keep port 8088 LAN-only unless access control is added.

For a manual systemd unit, use `Restart=no` and leave it disabled by default
so loading the experimental model is an explicit operator action. Never put
model paths, tokens, or credentials in the unit's environment.

## Benchmark reference

The benchmark uses one warmup followed by three measured streamed requests for
each workload, with temperature 0 and Qwen thinking disabled. Rates are the
median of per-request measurements; peak memory is the maximum unique
amdgpu-client `vram + gtt` allocation sampled during the measured requests.
On unified memory this is **workload allocation**, not installed RAM or a
fixed GPU aperture.

Historical LocalMaxxing run (throughput only; semantically invalid):

| Metric | Result |
|---|---:|
| Decode | **179.134 tok/s** |
| Prefill | **501.079 tok/s** |
| Total | **460.201 tok/s** |
| TTFT | **5,091.108 ms** |
| Peak workload allocation | **7.118 GB** |

The representative workload was 2,549 prompt tokens and 128 output tokens.
All measured outputs had the same hash and the n-gram draft accepted 125/125
tokens, but later semantic testing showed that hash equality was a false
correctness oracle: the accelerated profiles reproduced the same degenerate
output. The public LocalMaxxing record is
`cmtbd3r5g0026qq012qf9p1xp`.

## What did not win

- Whole-layer dual-device ROCm splitting was slower than 8060S-only because
  per-step synchronization dominated decode.
- Vulkan layer splitting was slower on the long workload than the ROCm
  `--n-cpu-moe 64` profile.
- Vulkan row splitting fails on the relevant backend with
  `device Vulkan0 does not support split buffers`.
- The current Flash-Next GGUF does not expose a usable MTP head; `ngram-mod`
  is the supported speculative path.

### Selective R9700 expert execution

The Qwen4Exp llama.cpp runtime can place complete routed expert stacks for
selected layers on the R9700 with `--override-tensor`, while a later
`--n-cpu-moe 64` fallback keeps every other routed layer on CPU. Device
ordering must be unmasked:

- `ROCm0` = Radeon AI PRO R9700 (`gfx1201`)
- `ROCm1` = Radeon 8060S (`gfx1151`)

Build the runtime for both architectures and use
[`scripts/run-qwen4exp-selective-experts.sh`](../scripts/run-qwen4exp-selective-experts.sh).
The tested 20-layer profile placed layers 0 through 19 on the R9700:

| Profile | Long decode | Long prefill | R9700 allocation | Output hash |
|---|---:|---:|---:|---|
| CPU MoE 64 (winner) | **179.134 tok/s** | **501.079 tok/s** | 0 GB | `a85b737…` |
| R9700 layers 0-19 | 153.529 tok/s | 421.716 tok/s | 30.385 GB | `a85b737…` |
| R9700 layers 0-3 | 149.762 tok/s | 397.611 tok/s | 10.950 GB | `a85b737…` |

The benchmark used the same 2,549-token prompt, 128 generated tokens,
temperature zero, one warmup, and three measured runs. All three profiles
produced the exact same SHA-256 output hash, but this is not semantic
correctness evidence because the ROCm control itself is degenerate. The R9700 values above are that
device's unique amdgpu client allocation, not combined system/GPU capacity.

The result closes the coarse scheduler experiment: moving fewer complete
expert stacks does not recover the CPU-MoE baseline. The bottleneck is the
per-layer cross-device scheduler boundary, not insufficient R9700 capacity
or incorrect Qwen4Exp state ownership.

## Engineering conclusion

The validated byte-placement split is:

- 8060S: HC/GDN/QSA graph, recurrent and index caches, shared experts, and
  protected core tensors;
- CPU: the 28.80 GB PLE gather table and routed stacks not selected for the
  R9700;
- R9700: only explicitly selected routed `ffn_*_exps` tensor stacks.

The native upstream Qwen4Exp graph remains the implementation source, but the
current ROCm execution path is not a valid numerical oracle. Copying
it into a second Lucebox backend would duplicate model and cache semantics
without changing the measured transfer boundary. A future attempt should
therefore start below ggml's whole-tensor scheduler: batch only active expert
contributions, overlap R9700 work with primary-device shared-expert work, and
join one compact activation result per layer. Do not move GDN/recurrent state,
PLE history, QSA cache, or shared-expert state.

## Rollback / safety

- Stop the experimental server before starting another process on the same
  port.
- Keep the known-good Vulkan and normal Qwen3.8 launchers available separately.
- Do not use broad `pkill -f` patterns over SSH; stop the named systemd unit or
  target the exact binary PID.
- Do not expose the unauthenticated inference port to the public Internet.
- Model weights, benchmark logs containing private paths, and credentials are
  intentionally excluded from this repository.
