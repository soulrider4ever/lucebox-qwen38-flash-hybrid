#!/usr/bin/env bash
set -euo pipefail

# Semantically validated Qwen3.8 Flash-Next profile for OpenWebUI on Lucebox.
#
# This deliberately uses the 8060S Vulkan backend and disables Qwen4Exp graph
# reuse, speculative decoding, and default reasoning. The faster ROCm CPU-MoE
# benchmark profile can emit an endless stream of '/' tokens and must not be
# used as a chat service.

: "${LLAMA_SERVER:?set LLAMA_SERVER to the Vulkan llama-server binary}"
: "${MODEL:?set MODEL to split GGUF shard 1}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8088}
CTX_SIZE=${CTX_SIZE:-32768}
MODEL_ALIAS=${MODEL_ALIAS:-Qwen3.8-Flash-Next-IQ4_XS-ROCm-CPU-MoE64}

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "LLAMA_SERVER is not executable: $LLAMA_SERVER" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "MODEL shard does not exist: $MODEL" >&2
    exit 2
fi

devices=$("$LLAMA_SERVER" --list-devices 2>&1)
grep -q 'Vulkan1: Radeon 8060S' <<<"$devices" || {
    echo "Vulkan1 is not the expected Radeon 8060S; refusing unsafe placement" >&2
    printf '%s\n' "$devices" >&2
    exit 3
}

export LLAMA_GRAPH_REUSE_DISABLE=1

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --device Vulkan1 \
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
