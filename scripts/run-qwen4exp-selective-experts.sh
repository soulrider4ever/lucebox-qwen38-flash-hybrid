#!/usr/bin/env bash
set -euo pipefail

# Run Qwen3.8 Flash-Next with the protected Qwen4Exp graph on the 8060S,
# selected complete routed-expert layer stacks on the R9700, and all remaining
# routed stacks on CPU.
#
# Required:
#   LLAMA_SERVER=/path/to/dual-architecture/llama-server
#   MODEL=/path/to/Qwen3.8-Flash-Next-...-00001-of-00003.gguf
#
# Optional:
#   R9700_LAYERS=20  HOST=127.0.0.1  PORT=18087  CTX_SIZE=32768

: "${LLAMA_SERVER:?set LLAMA_SERVER to the Qwen4Exp llama-server binary}"
: "${MODEL:?set MODEL to split GGUF shard 1}"

R9700_LAYERS=${R9700_LAYERS:-20}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18087}
CTX_SIZE=${CTX_SIZE:-32768}
ROCM_LIB_PATH=${ROCM_LIB_PATH:-/opt/rocm/core-7.14/lib:/opt/rocm/lib}

if ! [[ "$R9700_LAYERS" =~ ^[0-9]+$ ]] ||
   (( R9700_LAYERS < 1 || R9700_LAYERS > 48 )); then
    echo "R9700_LAYERS must be an integer from 1 through 48" >&2
    exit 2
fi
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

layers=$(seq -s '|' 0 "$((R9700_LAYERS - 1))")
override="blk\\.(${layers})\\.ffn_(gate|up|down)_exps\\.weight=ROCm0"

exec "$LLAMA_SERVER" +    -m "$MODEL" +    --device ROCm1,ROCm0 +    --tensor-split 1,0 +    --split-mode layer +    --override-tensor "$override" +    --n-cpu-moe 64 +    --n-gpu-layers 99 +    --ctx-size "$CTX_SIZE" +    --threads 32 +    --threads-batch 32 +    --ubatch-size 1024 +    --host "$HOST" +    --port "$PORT" +    --jinja +    --flash-attn on +    --no-ui +    --load-mode none +    --spec-type ngram-mod +    --alias "Qwen3.8-Flash-Next-IQ4_XS-Hybrid-R9700-L0-$((R9700_LAYERS - 1))"
