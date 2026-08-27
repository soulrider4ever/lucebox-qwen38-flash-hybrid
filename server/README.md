# Qwen3.8 Flash-Next server build

This directory contains the CMake build for the Lucebox runtime used by the
Qwen3.8 Flash-Next fork. The model-serving executable is the vendored
`server/deps/llama.cpp/examples/server/llama-server`; the native Lucebox
server and Qwen4Exp validation targets are built from this directory.

## ROCm build

Use ROCm 7.x and compile both device architectures used by the validated
R9700/Strix Halo profile:

```bash
cmake -S server -B server/build-hip-dual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1201;gfx1151' \
  -DDFLASH27B_ENABLE_BSA=OFF
cmake --build server/build-hip-dual --target dflash_server test_server_unit -j$(nproc)
```

The Qwen4Exp validation targets are:

```bash
cmake --build server/build-hip-dual --target \
  test_qwen4exp_hybrid_plan test_qwen4exp_gguf_inventory \
  test_qwen4exp_expert_shards test_qwen4exp_expert_materializer \
  test_qwen4exp_core_loader
```

## Model

The model is distributed separately as three GGUF shards. Follow the
model-specific [runbook](../docs/qwen38-flash-next-lucebox.md) for download,
launcher, split placement, and semantic verification instructions.
