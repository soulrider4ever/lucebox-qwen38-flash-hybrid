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
