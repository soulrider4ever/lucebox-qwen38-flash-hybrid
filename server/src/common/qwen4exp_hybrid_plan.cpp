#include "qwen4exp_hybrid_plan.h"

#include "moe_hybrid_routing_stats.h"

namespace dflash::common {

Qwen4ExpTensorRole classify_qwen4exp_tensor(const std::string & name) {
    const auto has = [&name](const char * needle) {
        return name.find(needle) != std::string::npos;
    };
    if (has("token_embd") || has("tok_embd")) return Qwen4ExpTensorRole::TokenEmbedding;
    if (has("output") || has("hc_head_")) return Qwen4ExpTensorRole::Output;
    if (has("hc_")) return Qwen4ExpTensorRole::HyperConnection;
    if (has("indexer_") || has("index_q_") || has("index_k_")) return Qwen4ExpTensorRole::SparseIndexer;
    if (has("ple_") || has("per_layer_token_embd")) return Qwen4ExpTensorRole::PleHistory;
    if (has("wqkv") || has("ssm_") || has("attn_gate")) return Qwen4ExpTensorRole::RecurrentMixer;
    if (has("attn_q") || has("attn_k") || has("attn_v") || has("attn_out")) return Qwen4ExpTensorRole::FullAttention;
    if (has("ffn_gate_inp_shexp") || has("ffn_gate_shexp") || has("ffn_up_shexp") || has("ffn_down_shexp")) return Qwen4ExpTensorRole::SharedExpert;
    if (has("ffn_gate_inp")) return Qwen4ExpTensorRole::ExpertRouter;
    if (has("ffn_gate_exps") || has("ffn_up_exps") || has("ffn_down_exps") || has("ffn_gate_up_exps")) return Qwen4ExpTensorRole::RoutedExpert;
    return Qwen4ExpTensorRole::Unknown;
}

bool Qwen4ExpHybridPlan::valid(std::string * error) const {
    const auto fail = [error](const char * message) {
        if (error) *error = message;
        return false;
    };
    if (config.n_layer <= 0 || config.n_expert <= 0 || config.n_expert_used <= 0) return fail("invalid Qwen4Exp dimensions");
    if (config.n_expert_used > config.n_expert) return fail("n_expert_used exceeds n_expert");
    if (hyperconnection_owner != config.primary_device || recurrent_state_owner != config.primary_device ||
        sparse_index_owner != config.primary_device || ple_history_owner != config.primary_device ||
        shared_expert_owner != config.primary_device) return fail("non-routed Qwen4Exp state escaped the primary owner");
    if (!experts.matches(config.n_layer, config.n_expert, config.n_expert_used)) return fail("expert placement dimensions do not match Qwen4Exp");
    return true;
}

bool build_qwen4exp_hybrid_plan(
    const Qwen4ExpHybridConfig & config,
    const MoeHybridRoutingStats & routing,
    const std::vector<uint64_t> & layer_expert_bytes,
    const std::vector<uint64_t> & layer_primary_fixed_bytes,
    uint64_t primary_expert_budget_bytes,
    double primary_to_peer_rate,
    Qwen4ExpHybridPlan & out,
    std::string * error) {
    if (config.n_layer <= 0 || config.n_expert <= 0 || config.n_expert_used <= 0) {
        if (error) *error = "invalid Qwen4Exp dimensions";
        return false;
    }
    if ((int) layer_expert_bytes.size() != config.n_layer ||
        (int) layer_primary_fixed_bytes.size() != config.n_layer) {
        if (error) *error = "placement vectors must cover every layer";
        return false;
    }
    if (!(primary_to_peer_rate > 0.0)) {
        if (error) *error = "primary_to_peer_rate must be positive";
        return false;
    }

    Qwen4ExpHybridPlan plan;
    plan.config = config;
    plan.hyperconnection_owner = config.primary_device;
    plan.recurrent_state_owner = config.primary_device;
    plan.sparse_index_owner = config.primary_device;
    plan.ple_history_owner = config.primary_device;
    plan.shared_expert_owner = config.primary_device;

    MoeHybridCriticalPathConfig critical;
    critical.active_experts = config.n_expert_used;
    critical.main_to_peer_rate = primary_to_peer_rate;
    if (!MoeHybridPlacement::build_critical_path_balanced_from_stats(
            routing, layer_expert_bytes, layer_primary_fixed_bytes,
            primary_expert_budget_bytes, critical, plan.experts, error)) return false;
    if (!plan.valid(error)) return false;
    out = std::move(plan);
    return true;
}

} // namespace dflash::common
