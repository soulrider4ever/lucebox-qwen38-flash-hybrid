// GGUF-facing Qwen4Exp inventory.
//
// This is intentionally independent of the eventual execution backend: it
// reads tensor metadata, validates the routed-expert surface, and reports the
// byte budget available to a hybrid placement planner. It never allocates or
// moves a tensor.
#pragma once

#include "qwen4exp_hybrid_plan.h"

#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen4ExpLayerExpertInventory {
    uint64_t total_bytes = 0;
    uint64_t gate_bytes = 0;
    uint64_t up_bytes = 0;
    uint64_t down_bytes = 0;
    uint32_t tensor_count = 0;
    uint32_t expert_tensor_count = 0;
};

struct Qwen4ExpGgufInventory {
    std::string architecture;
    int n_layer = 0;
    int first_routed_layer = -1;
    int n_expert = 0;
    int n_expert_used = 0;
    std::vector<Qwen4ExpLayerExpertInventory> layers;
    Qwen4ExpTensorInventory tensor_roles;

    bool valid(std::string * error = nullptr) const;
};

// Scan the tensor metadata in an already-open GGUF context. gguf_context is
// read-only and remains owned by the caller. The scan rejects malformed block
// names, duplicate routed surfaces, and a partial gate/up/down expert set.
bool scan_qwen4exp_gguf_inventory(const gguf_context * gguf,
                                  Qwen4ExpGgufInventory & out,
                                  std::string * error = nullptr);

} // namespace dflash::common
