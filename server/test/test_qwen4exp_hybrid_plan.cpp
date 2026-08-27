#include "qwen4exp_hybrid_plan.h"
#include "qwen4exp_gguf_inventory.h"
#include "moe_hybrid_routing_stats.h"

#include <cassert>
#include <string>

using namespace dflash::common;

int main() {
    std::string error;
    assert(classify_qwen4exp_tensor("blk.0.hc_attn_down.weight") == Qwen4ExpTensorRole::HyperConnection);
    assert(classify_qwen4exp_tensor("blk.0.ssm_beta.weight") == Qwen4ExpTensorRole::RecurrentMixer);
    assert(classify_qwen4exp_tensor("blk.0.indexer_q_proj.weight") == Qwen4ExpTensorRole::SparseIndexer);
    assert(classify_qwen4exp_tensor("blk.0.ffn_gate_inp.weight") == Qwen4ExpTensorRole::ExpertRouter);
    assert(classify_qwen4exp_tensor("blk.0.ffn_down_exps.weight") == Qwen4ExpTensorRole::RoutedExpert);
    assert(classify_qwen4exp_tensor("blk.0.ffn_down_shexp.weight") == Qwen4ExpTensorRole::SharedExpert);

    const auto routed = identify_qwen4exp_tensor("blk.17.ffn_down_exps.weight");
    assert(routed.valid());
    assert(routed.layer == 17);
    assert(routed.expert_stacked);
    assert(qwen4exp_tensor_may_move_to_peer(routed));

    const auto shared = identify_qwen4exp_tensor("blk.17.ffn_down_shexp.weight");
    assert(shared.valid());
    assert(shared.layer == 17);
    assert(!shared.expert_stacked);
    assert(!qwen4exp_tensor_may_move_to_peer(shared));

    const auto state = identify_qwen4exp_tensor("blk.17.ssm_beta.weight");
    assert(state.valid());
    assert(state.layer == 17);
    assert(!qwen4exp_tensor_may_move_to_peer(state));

    assert(!identify_qwen4exp_tensor("blk.bad.ffn_down_exps.weight").valid());
    assert(!identify_qwen4exp_tensor("blk.17").valid());

    Qwen4ExpTensorInventory inventory;
    assert(inventory.observe("blk.0.ssm_beta.weight"));
    assert(inventory.observe("blk.0.ffn_down_exps.weight"));
    assert(inventory.observe("blk.1.ffn_down_exps.weight"));
    assert(inventory.validate(2, 0, &error));
    assert(!inventory.observe("blk.bad.ffn_down_exps.weight", &error));
    assert(error.find("malformed") != std::string::npos);

    Qwen4ExpTensorInventory missing;
    assert(missing.observe("blk.0.ffn_down_exps.weight"));
    assert(!missing.validate(2, 0, &error));
    assert(error.find("layer 1") != std::string::npos);

    Qwen4ExpHybridPlan plan;
    plan.config.n_layer = 2;
    plan.config.n_expert = 4;
    plan.config.n_expert_used = 2;
    plan.experts.n_layer = 2;
    plan.experts.n_expert = 4;
    plan.experts.n_expert_used = 2;
    plan.experts.hot_counts = {1, 1};
    plan.experts.hot_expert_ids = {{0}, {1}};
    assert(plan.valid(&error));
    plan.recurrent_state_owner = 1;
    assert(!plan.valid(&error));
    assert(error.find("non-routed") != std::string::npos);

    Qwen4ExpGgufInventory gguf;
    gguf.architecture = "qwen4exp";
    gguf.n_layer = 2;
    gguf.n_expert = 4;
    gguf.n_expert_used = 2;
    gguf.first_routed_layer = 0;
    gguf.layers.resize(2);
    for (auto & layer : gguf.layers) {
        layer.total_bytes = 100;
        layer.per_expert_bytes = 25;
        layer.gate_bytes = 32;
        layer.up_bytes = 32;
        layer.down_bytes = 36;
        layer.gate_per_expert_bytes = 8;
        layer.up_per_expert_bytes = 8;
        layer.down_per_expert_bytes = 9;
        layer.expert_tensor_count = 3;
    }
    MoeHybridRoutingStats routing;
    assert(routing.init(2, 4, 2));
    Qwen4ExpHybridPlan from_inventory;
    assert(build_qwen4exp_hybrid_plan_from_inventory(
        gguf, routing, {0, 0}, 150, 1.0, from_inventory, &error));
    assert(from_inventory.valid(&error));
    assert(from_inventory.experts.n_layer == 2);
    assert(from_inventory.experts.n_expert == 4);
    return 0;
}
