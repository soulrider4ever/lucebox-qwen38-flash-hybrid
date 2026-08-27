#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8088}

curl -fsS "$BASE_URL/health" | jq -e '.status == "ok"' >/dev/null
model=${MODEL_ALIAS:-$(curl -fsS "$BASE_URL/v1/models" | jq -er '.data[0].id')}

request() {
    local prompt=$1
    local max_tokens=$2
    jq -nc \
        --arg model "$model" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$max_tokens" \
        '{model:$model,messages:[{role:"user",content:$prompt}],temperature:0,max_tokens:$max_tokens,stream:false}' |
        curl -fsS --max-time 180 \
            "$BASE_URL/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            --data-binary @-
}

greeting=$(request 'hi' 64 | jq -er '.choices[0].message.content')
if [[ ! "$greeting" =~ [[:alpha:]] ]]; then
    echo "semantic canary failed: greeting contains no letters: $greeting" >&2
    exit 1
fi

factual=$(request 'Reply with exactly: Buffalo is in New York.' 32 | jq -er '.choices[0].message.content')
if [[ "$factual" != 'Buffalo is in New York.' ]]; then
    echo "semantic canary failed: unexpected factual response: $factual" >&2
    exit 1
fi

printf 'OpenWebUI semantic canary passed\n'
printf 'model=%s\n' "$model"
printf 'greeting=%s\n' "$greeting"
