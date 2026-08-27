# Qwen3.8 Flash-Next Lucebox Runtime

This repository is a focused Lucebox fork for running **Qwen3.8
Flash-Next** on the AMD Radeon AI PRO R9700 + Ryzen AI MAX+ 395 (Strix Halo)
system. It contains the Qwen4Exp loader/materialization work, the validated
dual-ROCm launcher, and the model-specific runbook.

## Validated profile

- Model: `Qwen3.8-Flash-Next-UD-IQ4_XS` (three GGUF shards)
- Devices: `ROCm0` = R9700, `ROCm1` = Radeon 8060S
- Placement: `--tensor-split 0.45,0.55`, layer split, 8060S main GPU
- Context: 32K
- Attention: dense below 32K; QSA is retained above the deployment window
- Graph reuse on; CPU MoE and speculation off
- Measured: 25.639 tok/s decode, 537.394 tok/s prefill

The benchmark result and limitations are documented in
[`docs/qwen38-flash-next-lucebox.md`](docs/qwen38-flash-next-lucebox.md).

## Build

Clone with the llama.cpp submodule:

```bash
git clone --recurse-submodules https://github.com/soulrider4ever/lucebox-qwen38-flash-hybrid.git
cd lucebox-qwen38-flash-hybrid
cmake -S server -B server/build-hip-dual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1201;gfx1151' \
  -DDFLASH27B_ENABLE_BSA=OFF
cmake --build server/build-hip-dual --target dflash_server test_server_unit -j$(nproc)
```

The vendored `server/deps/llama.cpp` build provides the `llama-server`
executable used by the Qwen launcher. Build it with ROCm support for both
target architectures.

## Run

```bash
MODEL=/models/qwen38-flash-next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
LLAMA_SERVER=/path/to/llama-server \
scripts/run-qwen4exp-dual-rocm-safe.sh
```

For the validated 8060S Vulkan rollback profile, use
`scripts/run-qwen4exp-openwebui-safe.sh`. The semantic smoke test is
`scripts/test-qwen4exp-openwebui.sh`.

## Repository scope

The repository intentionally excludes unrelated model integrations,
optimization experiments, client harnesses, benchmark suites, and their
documentation. Shared llama.cpp/ggml and server infrastructure remains where
it is required to build the Qwen runtime.

## License

The project is distributed under the Apache License 2.0; see [LICENSE](LICENSE).
