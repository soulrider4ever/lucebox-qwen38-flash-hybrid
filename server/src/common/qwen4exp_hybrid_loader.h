// Loader-facing preparation for a Qwen4Exp hybrid placement.
//
// This deliberately stops before model execution: it opens split GGUF
// metadata, validates the routed surface, and returns a plan whose byte
// accounting can be consumed by a future expert storage backend.
#pragma once

#include "qwen4exp_hybrid_plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

bool prepare_qwen4exp_hybrid_plan_from_gguf(
    const std::vector<std::string> & shard_paths,
    const MoeHybridRoutingStats & routing,
    const std::vector<uint64_t> & layer_primary_fixed_bytes,
    uint64_t primary_expert_budget_bytes,
    double primary_to_peer_rate,
    Qwen4ExpHybridPlan & out,
    std::string * error = nullptr);

} // namespace dflash::common
