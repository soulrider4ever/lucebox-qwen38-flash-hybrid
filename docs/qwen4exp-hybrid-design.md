# Qwen4Exp hybrid placement design

## Why this is a separate adapter

Qwen4Exp (the architecture behind Qwen3.8-Flash-Next) is not a DeepSeek4
target with different tensor names. Its graph carries four residual streams
through low-rank hyper-connections, mixes full-attention and recurrent/GDN
layers, and may maintain PLE n-gram history plus a QSA sparse index cache.
Those tensors and states must remain on one coherent target owner.

The only safe first seam for Lucebox's asymmetric runtime is the routed MoE
FFN. The planner in `src/common/qwen4exp_hybrid_plan.*` therefore enforces:

- hyper-connection weights and residual state stay on the primary owner;
- recurrent/GDN state stays on the primary owner;
- QSA indexer and PLE history stay on the primary owner;
- the shared expert stays on the primary owner;
- only `ffn_*_exps` routed experts enter hot/cold placement.

This is deliberately a planning seam, not a fake Qwen4Exp backend. The full
Qwen4Exp loader/graph/cache implementation from llama.cpp still needs to be
ported before this plan can drive inference.

## Adapter boundary (current implementation)

`server/src/common/qwen4exp_hybrid_plan.*` now provides a strict tensor
identity parser for canonical `blk.<layer>.<tensor>[.weight]` names. It marks
only `ffn_*_exps` routed stacks as peer-movable. Shared experts, routers,
hyper-connections, recurrent/GDN tensors, sparse indexer tensors, and PLE
history are explicitly non-movable. Malformed block names are rejected rather
than silently assigned to a placement tier.

This is the first executable adapter boundary. It is intentionally not wired
into the inference graph yet: the Qwen4Exp graph must first expose its routed
expert tensors and state ownership through a model-specific loader. The unit
test covers routed-vs-shared/state classification and malformed names.

## Intended Lucebox route

For the R9700 + Strix Halo system, the eventual plan is:

1. Run the Qwen4Exp graph on the primary owner (initially the R9700 for a
   discrete-device experiment).
2. Keep the latency-critical routed experts hot on that owner using routing
   observations and the critical-path byte budget.
3. Keep the long-tail routed experts on the peer owner or host, grouped by
   expert id so transfers are batched rather than issued once per token.
4. Transfer only the selected routed contributions and join them before the
   Qwen4Exp hyper-connection combine.

The existing `MoeHybridPlacement` critical-path allocator is reused, but its
result is wrapped with Qwen4Exp ownership invariants so a later graph adapter
cannot accidentally move recurrent state or sparse-index tensors with the
expert payload.

## Current measured control

Before attempting the R9700 path, the 8060S-only ROCm control is useful:
`--n-cpu-moe 64` gave approximately 44.2 tok/s on the short count workload,
248.7 tok/s on natural prose, and 185.6 tok/s on the long synthetic workload,
with stable output hashes and roughly 7.1 GB GPU allocation. That result is
an n-gram/canreuse control, not evidence that Qwen4Exp hybrid offload is
complete; it gives the placement work a concrete performance target.

## Split-shard expert data boundary

`qwen4exp_expert_shards.*` is the loader boundary for the next runtime stage.
It opens every GGUF shard with `no_alloc`, merges the model metadata, and
records the absolute file offset and byte size of each layer's routed gate,
up, and down tensor. `Qwen4ExpExpertRegion::slice()` then derives a checked
contiguous range for one expert. The map is backend-neutral: a future R9700
materializer can mmap each shard and upload only selected slices, while the
Qwen4Exp graph keeps its recurrent state, sparse index cache, PLE history,
router, and shared expert on the primary owner.

This is intentionally not a runtime claim. Until the Qwen4Exp graph is wired
into Lucebox, the map and planner are validated preparation code only.
