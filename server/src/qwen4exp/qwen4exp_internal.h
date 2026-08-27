// Qwen4Exp model contract for the native Lucebox backend.
#pragma once

#include "common/qwen4exp_hybrid_plan.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dflash::common {

struct Qwen4ExpCoreTensor {
    std::string name;
    Qwen4ExpTensorRole role = Qwen4ExpTensorRole::Unknown;
    int layer = -1;
    int shard = -1;
    uint64_t file_offset = 0;
    uint64_t file_bytes = 0;
    ggml_tensor * tensor = nullptr;
};

struct Qwen4ExpCoreWeights {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;       // borrowed
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<Qwen4ExpCoreTensor> tensors;
    std::unordered_map<std::string, ggml_tensor *> by_name;
    uint64_t resident_bytes = 0;
    uint64_t routed_expert_bytes = 0;
    uint64_t ple_table_bytes = 0;
    int n_layer = 0;

    ggml_tensor * find(const std::string & name) const {
        const auto it = by_name.find(name);
        return it == by_name.end() ? nullptr : it->second;
    }
};

bool load_qwen4exp_protected_core_from_shards(
    const std::vector<std::string> & shard_paths,
    ggml_backend_t backend,
    Qwen4ExpCoreWeights & out,
    std::string * error = nullptr);

void free_qwen4exp_core_weights(Qwen4ExpCoreWeights & weights);

} // namespace dflash::common
