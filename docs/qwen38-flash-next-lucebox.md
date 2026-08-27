# Qwen3.8 Flash-Next on Lucebox

This document describes the experimental Qwen3.8-Flash-Next work in this
repository and the fastest validated Lucebox profile.

> **Status:** experimental, but correctness-checked on Lucebox. The current
> winner is ROCm on the Strix Halo 8060S with CPU-resident routed MoE layers.
> This is not yet true R9700 expert offload.

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

This builds the Lucebox fork and its placement/test infrastructure. The
currently validated Flash-Next server binary was built from a separate
Qwen4Exp-enabled llama.cpp checkout because the fork's Qwen4Exp adapter is
not yet a complete model loader. Do not represent the planner seam as a
drop-in replacement for that binary.

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

## Fastest validated profile

On the reference Lucebox, the working command was:

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

Latest validated LocalMaxxing run:

| Metric | Result |
|---|---:|
| Decode | **179.134 tok/s** |
| Prefill | **501.079 tok/s** |
| Total | **460.201 tok/s** |
| TTFT | **5,091.108 ms** |
| Peak workload allocation | **7.118 GB** |

The representative workload was 2,549 prompt tokens and 128 output tokens.
All measured outputs had the same hash and the n-gram draft accepted 125/125
tokens. The public LocalMaxxing record is
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

## Development direction

The next real milestone is expert-level remote execution on the R9700:
batch selected expert contributions, transfer only routed results, and join
them before Qwen4Exp's hyper-connection combine. Do not move GDN/recurrent
state, PLE history, QSA cache, or shared-expert state as if they were ordinary
DeepSeek tensors. Every implementation step must pass correctness tests and
the same three-workload benchmark before it is considered an improvement.

## Rollback / safety

- Stop the experimental server before starting another process on the same
  port.
- Keep the known-good Vulkan and normal Qwen3.8 launchers available separately.
- Do not use broad `pkill -f` patterns over SSH; stop the named systemd unit or
  target the exact binary PID.
- Do not expose the unauthenticated inference port to the public Internet.
- Model weights, benchmark logs containing private paths, and credentials are
  intentionally excluded from this repository.
