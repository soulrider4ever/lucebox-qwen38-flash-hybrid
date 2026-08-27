#include "qwen4exp_hybrid_plan.h"

#include <cassert>
#include <string>

using namespace dflash::common;

int main() {
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
    std::string error;
    assert(plan.valid(&error));
    plan.recurrent_state_owner = 1;
    assert(!plan.valid(&error));
    assert(error.find("non-routed") != std::string::npos);
    return 0;
}
