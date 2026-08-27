#include "common/qwen4exp_gguf_inventory.h"
#include "common/qwen4exp_hybrid_loader.h"
#include "common/moe_hybrid_routing_stats.h"

#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL-SHARD [MODEL-SHARD ...]\n", argv[0]);
        return 2;
    }
    gguf_init_params params{};
    params.no_alloc = true;
    std::vector<gguf_context *> contexts;
    for (int i = 1; i < argc; ++i) {
        gguf_context * context = gguf_init_from_file(argv[i], params);
        if (!context) {
            std::fprintf(stderr, "failed to open GGUF: %s\n", argv[i]);
            for (gguf_context * prior : contexts) gguf_free(prior);
            return 1;
        }
        contexts.push_back(context);
    }
    std::vector<const gguf_context *> shards(contexts.begin(), contexts.end());
    dflash::common::Qwen4ExpGgufInventory inventory;
    std::string error;
    const bool ok = dflash::common::scan_qwen4exp_gguf_shards(shards, inventory, &error);
    if (ok) {
        std::printf("arch=%s layers=%d experts=%d used=%d first_moe=%d tensors=%lld\n",
                    inventory.architecture.c_str(), inventory.n_layer,
                    inventory.n_expert, inventory.n_expert_used,
                    inventory.first_routed_layer,
                    static_cast<long long>(shards.size()));
        for (int layer = inventory.first_routed_layer; layer < inventory.n_layer; ++layer) {
            const auto & row = inventory.layers[static_cast<size_t>(layer)];
            std::printf("layer=%d expert_bytes=%llu per_expert=%llu gate=%llu up=%llu down=%llu tensors=%u\n",
                        layer,
                        static_cast<unsigned long long>(row.total_bytes),
                        static_cast<unsigned long long>(row.per_expert_bytes),
                        static_cast<unsigned long long>(row.gate_bytes),
                        static_cast<unsigned long long>(row.up_bytes),
                        static_cast<unsigned long long>(row.down_bytes),
                        row.expert_tensor_count);
        }
        assert(row.per_expert_bytes > 0);
    } else {
        std::fprintf(stderr, "inventory rejected: %s\n", error.c_str());
    }
    if (ok) {
        dflash::common::MoeHybridRoutingStats routing;
        assert(routing.init(inventory.n_layer, inventory.n_expert, inventory.n_expert_used));
        dflash::common::Qwen4ExpHybridPlan plan;
        std::vector<uint64_t> fixed((size_t) inventory.n_layer, 0);
        assert(dflash::common::prepare_qwen4exp_hybrid_plan_from_gguf(
            std::vector<std::string>(argv + 1, argv + argc), routing, fixed,
            512ULL * 1024ULL * 1024ULL, 1.0, plan, &error));
        assert(plan.valid(&error));
    }
    for (gguf_context * context : contexts) gguf_free(context);
    return ok ? 0 : 1;
}
