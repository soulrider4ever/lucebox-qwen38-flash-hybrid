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

This repository does not duplicate the complete Qwen4Exp inference graph.
Instead, generation uses the native Qwen4Exp llama.cpp graph as the numerical
source of truth while this adapter supplies two tested ownership boundaries:
selected routed experts can be materialized from split GGUF shards into
Lucebox's common MoE storage, and the protected non-expert core can be loaded
on the primary GPU. The native graph's HC/GDN/QSA/PLE caches remain intact.

## Adapter boundary (current implementation)

`server/src/common/qwen4exp_hybrid_plan.*` now provides a strict tensor
identity parser for canonical `blk.<layer>.<tensor>[.weight]` names. It marks
only `ffn_*_exps` routed stacks as peer-movable. Shared experts, routers,
hyper-connections, recurrent/GDN tensors, sparse indexer tensors, and PLE
history are explicitly non-movable. Malformed block names are rejected rather
than silently assigned to a placement tier.

This is the first executable adapter boundary. The unit test covers
routed-vs-shared/state classification and malformed names. For token
generation, the runtime integration maps the same routed tensor names through
llama.cpp's `--override-tensor` facility and leaves all protected tensors on
the 8060S graph owner.

## Selective expert-stack route

For the R9700 + Strix Halo system, the validated coarse route is:

1. Run the protected Qwen4Exp graph and state caches on the 8060S.
2. Keep complete routed expert stacks for selected layers on the R9700.
3. Keep all unselected routed stacks on CPU.
4. Let ggml schedule the routed MoE operations and cross-device joins.

This route produced exact baseline output hashes but lost to CPU MoE at both
four and twenty R9700-resident layers. The next-level design, if revisited,
must operate below the whole-tensor scheduler: move only active experts, batch
their contributions, and overlap remote work with the shared expert.

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

The map and planner are backend-neutral preparation code. Runtime generation
is validated separately through the native Qwen4Exp llama.cpp graph and the
topology-checking launcher documented in
`qwen38-flash-next-lucebox.md`.

## R9700 materialization canaries

The split-shard path has now been exercised against the real three-file
Qwen3.8-Flash-Next UD-IQ4_XS model on HIP device 0 (Radeon AI PRO R9700):

- one exact routed expert per layer was uploaded across all 48 layers;
- the resulting R9700 allocation was 116,263,168 bytes;
- a 64 KiB device readback matched the source GGUF bytes exactly;
- the protected core loader bound 1,079 tensors and uploaded 5,351,626,240
  bytes to the R9700;
- it excluded 59,519,795,200 routed-expert bytes for hybrid placement and
  28,800,138,240 PLE gather-table bytes for CPU ownership.

Both canaries released their allocations after validation. Model correctness
and inference speed were then tested separately with selective full-stack
placement: the four-layer and twenty-layer R9700 profiles reproduced the
baseline output hashes exactly but were slower than CPU MoE. The 8060S-only
baseline was restored after the campaign.
