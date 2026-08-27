#include "qwen4exp_hybrid_loader.h"

#include "qwen4exp_gguf_inventory.h"

#include "gguf.h"

#include <memory>

namespace dflash::common {

namespace {

struct GgufContextDeleter {
    void operator()(gguf_context * context) const {
        if (context) gguf_free(context);
    }
};

void fail(std::string * error, const std::string & message) {
    if (error) *error = message;
}

} // namespace

bool prepare_qwen4exp_hybrid_plan_from_gguf(
    const std::vector<std::string> & shard_paths,
    const MoeHybridRoutingStats & routing,
    const std::vector<uint64_t> & layer_primary_fixed_bytes,
    uint64_t primary_expert_budget_bytes,
    double primary_to_peer_rate,
    Qwen4ExpHybridPlan & out,
    std::string * error) {
    out = {};
    if (shard_paths.empty()) {
        fail(error, "no Qwen4Exp GGUF shard paths supplied");
        return false;
    }

    gguf_init_params params{};
    params.no_alloc = true;
    std::vector<std::unique_ptr<gguf_context, GgufContextDeleter>> owned;
    std::vector<const gguf_context *> shards;
    owned.reserve(shard_paths.size());
    shards.reserve(shard_paths.size());
    for (const std::string & path : shard_paths) {
        if (path.empty()) {
            fail(error, "empty Qwen4Exp GGUF shard path");
            return false;
        }
        std::unique_ptr<gguf_context, GgufContextDeleter> context(
            gguf_init_from_file(path.c_str(), params));
        if (!context) {
            fail(error, "failed to open Qwen4Exp GGUF shard: " + path);
            return false;
        }
        shards.push_back(context.get());
        owned.push_back(std::move(context));
    }

    Qwen4ExpGgufInventory inventory;
    if (!scan_qwen4exp_gguf_shards(shards, inventory, error)) return false;
    return build_qwen4exp_hybrid_plan_from_inventory(
        inventory, routing, layer_primary_fixed_bytes,
        primary_expert_budget_bytes, primary_to_peer_rate, out, error);
}

} // namespace dflash::common
