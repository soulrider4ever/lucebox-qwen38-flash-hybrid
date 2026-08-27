#include "qwen4exp/qwen4exp_internal.h"

#include "ggml-cuda.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {
struct BackendGuard {
    ggml_backend_t backend = nullptr;
    ~BackendGuard() { if (backend) ggml_backend_free(backend); }
};
struct CoreGuard {
    Qwen4ExpCoreWeights weights;
    ~CoreGuard() { free_qwen4exp_core_weights(weights); }
};
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_SHARD [MODEL_SHARD ...]\n", argv[0]);
        return 2;
    }
    std::vector<std::string> shards(argv + 1, argv + argc);
    BackendGuard backend{ggml_backend_cuda_init(0)};
    if (!backend.backend) {
        std::fprintf(stderr, "failed to initialize HIP device 0\n");
        return 1;
    }
    CoreGuard core;
    std::string error;
    if (!load_qwen4exp_protected_core_from_shards(
            shards, backend.backend, core.weights, &error)) {
        std::fprintf(stderr, "core load failed: %s\n", error.c_str());
        return 1;
    }
    if (core.weights.n_layer != 48 ||
        !core.weights.find("token_embd.weight") ||
        !core.weights.find("blk.47.hc_ffn_inject.weight") ||
        core.weights.find("blk.0.ffn_gate_exps.weight") ||
        core.weights.find("per_layer_token_embd.weight")) {
        std::fprintf(stderr, "protected core ownership boundary is incorrect\n");
        return 1;
    }
    std::printf("PASS device=0 layers=%d tensors=%zu resident=%llu experts_skipped=%llu ple_skipped=%llu\n",
                core.weights.n_layer, core.weights.tensors.size(),
                static_cast<unsigned long long>(core.weights.resident_bytes),
                static_cast<unsigned long long>(core.weights.routed_expert_bytes),
                static_cast<unsigned long long>(core.weights.ple_table_bytes));
    return 0;
}
