// Qwen4Exp routed-expert materialization for split GGUF models.
//
// This is the first runtime-facing adapter boundary: it consumes the validated
// shard map and copies selected quantized expert slices into Lucebox's common
// MoE storage.  It deliberately does not build the Qwen4Exp attention/state
// graph; callers may use it for isolated load canaries before graph execution
// is enabled.
#pragma once

#include "moe_hybrid_storage.h"
#include "qwen4exp_expert_shards.h"

#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpExpertMaterializeConfig {
    // Owner of placement.hot_expert_ids.
    ggml_backend_t primary_backend = nullptr;

    // Optional owner of the complement.  When null, only primary experts are
    // allocated; this is useful for a bounded R9700 load/copy canary.
    ggml_backend_t peer_backend = nullptr;
    bool materialize_peer = false;
};

bool materialize_qwen4exp_experts_from_shards(
    const std::vector<std::string> & shard_paths,
    const MoeHybridPlacement & placement,
    const Qwen4ExpExpertMaterializeConfig & config,
    MoeHybridStorage & out,
    std::string * error = nullptr);

} // namespace dflash::common
