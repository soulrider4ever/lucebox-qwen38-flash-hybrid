// Qwen4Exp-specific boundary for Lucebox hybrid expert placement.
//
// Qwen4Exp cannot be treated as a DeepSeek4 clone: its hyper-connections,
// recurrent/GDN state, PLE history, and sparse index cache form one coherent
// target graph. This planning layer limits cross-device movement to routed
// expert tensors. It does not claim that the full Qwen4Exp graph is already
// implemented in Lucebox.
#pragma once

#include "moe_hybrid_placement.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

enum class Qwen4ExpTensorRole {
    Unknown, TokenEmbedding, Output, HyperConnection, RecurrentMixer,
    FullAttention, SparseIndexer, PleHistory, ExpertRouter, RoutedExpert,
    SharedExpert,
};

// Classify canonical llama.cpp Qwen4Exp tensor names. Unknown tensors stay on
// the target owner when a future loader consumes this boundary.
Qwen4ExpTensorRole classify_qwen4exp_tensor(const std::string & name);

struct Qwen4ExpHybridConfig {
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0;
    int primary_device = 0;       // latency-critical owner (future R9700)
    int peer_device = 1;          // complementary owner (future 8060S)
    bool allow_cpu_cold = true;
    bool move_shared_expert = false;
};

struct Qwen4ExpHybridPlan {
    Qwen4ExpHybridConfig config;
    MoeHybridPlacement experts;

    // Non-routed state is deliberately pinned to one owner. This prevents a
    // planner consumer from sending recurrent state or QSA/PLE tensors through
    // the expert transfer path.
    int hyperconnection_owner = 0;
    int recurrent_state_owner = 0;
    int sparse_index_owner = 0;
    int ple_history_owner = 0;
    int shared_expert_owner = 0;

    bool valid(std::string * error = nullptr) const;
};

// Build a critical-path-aware hot/cold plan from routing observations. The
// expert byte vector is per layer; non-expert tensors are not in the budget.
bool build_qwen4exp_hybrid_plan(
    const Qwen4ExpHybridConfig & config,
    const MoeHybridRoutingStats & routing,
    const std::vector<uint64_t> & layer_expert_bytes,
    const std::vector<uint64_t> & layer_primary_fixed_bytes,
    uint64_t primary_expert_budget_bytes,
    double primary_to_peer_rate,
    Qwen4ExpHybridPlan & out,
    std::string * error = nullptr);

} // namespace dflash::common
