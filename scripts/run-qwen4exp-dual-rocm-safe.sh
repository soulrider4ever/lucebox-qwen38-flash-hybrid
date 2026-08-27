#!/usr/bin/env bash
set -euo pipefail

# Semantically validated Qwen3.8 Flash-Next split profile for Lucebox.
#
# The complete model graph is split between the R9700 and 8060S. Do not add
# --n-cpu-moe: the CPU-routed MoE path is known to emit degenerate '/' output
# with this Qwen4Exp build.

: "${LLAMA_SERVER:?set LLAMA_SERVER to the dual-architecture HIP llama-server binary}"
: "${MODEL:?set MODEL to split GGUF shard 1}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8088}
CTX_SIZE=${CTX_SIZE:-32768}
MODEL_ALIAS=${MODEL_ALIAS:-Qwen3.8-Flash-Next-IQ4_XS-Split-R9700-8060S}
ROCM_LIB_PATH=${ROCM_LIB_PATH:-/opt/rocm/core-7.14/lib:/opt/rocm/lib}

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "LLAMA_SERVER is not executable: $LLAMA_SERVER" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "MODEL shard does not exist: $MODEL" >&2
    exit 2
fi

export LD_LIBRARY_PATH="$ROCM_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset HIP_VISIBLE_DEVICES
unset LLAMA_GRAPH_REUSE_DISABLE

devices=$("$LLAMA_SERVER" --list-devices 2>&1)
grep -q 'ROCm0: AMD Radeon AI PRO R9700' <<<"$devices" || {
    echo "ROCm0 is not the expected R9700; refusing unsafe placement" >&2
    printf '%s\n' "$devices" >&2
    exit 3
}
grep -q 'ROCm1: AMD Radeon 8060S' <<<"$devices" || {
    echo "ROCm1 is not the expected 8060S; refusing unsafe placement" >&2
    printf '%s\n' "$devices" >&2
    exit 3
}

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --device ROCm0,ROCm1 \
    --tensor-split 0.33,0.67 \
    --split-mode layer \
    --main-gpu 1 \
    --n-gpu-layers 99 \
    --ctx-size "$CTX_SIZE" \
    --threads 32 \
    --threads-batch 32 \
    --ubatch-size 1024 \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    --flash-attn on \
    --no-ui \
    --load-mode none \
    --spec-type none \
    --reasoning off \
    --alias "$MODEL_ALIAS"
