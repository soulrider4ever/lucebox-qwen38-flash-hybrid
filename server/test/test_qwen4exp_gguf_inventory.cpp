#include "common/qwen4exp_gguf_inventory.h"

#include "gguf.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL-SHARD\n", argv[0]);
        return 2;
    }
    gguf_init_params params{};
    params.no_alloc = true;
    gguf_context * context = gguf_init_from_file(argv[1], params);
    if (!context) {
        std::fprintf(stderr, "failed to open GGUF: %s\n", argv[1]);
        return 1;
    }
    dflash::common::Qwen4ExpGgufInventory inventory;
    std::string error;
    const bool ok = dflash::common::scan_qwen4exp_gguf_inventory(context, inventory, &error);
    if (ok) {
        std::printf("arch=%s layers=%d experts=%d used=%d first_moe=%d tensors=%lld\n",
                    inventory.architecture.c_str(), inventory.n_layer,
                    inventory.n_expert, inventory.n_expert_used,
                    inventory.first_routed_layer,
                    static_cast<long long>(gguf_get_n_tensors(context)));
        for (int layer = inventory.first_routed_layer; layer < inventory.n_layer; ++layer) {
            const auto & row = inventory.layers[static_cast<size_t>(layer)];
            std::printf("layer=%d expert_bytes=%llu gate=%llu up=%llu down=%llu tensors=%u\n",
                        layer,
                        static_cast<unsigned long long>(row.total_bytes),
                        static_cast<unsigned long long>(row.gate_bytes),
                        static_cast<unsigned long long>(row.up_bytes),
                        static_cast<unsigned long long>(row.down_bytes),
                        row.expert_tensor_count);
        }
    } else {
        std::fprintf(stderr, "inventory rejected: %s\n", error.c_str());
    }
    gguf_free(context);
    return ok ? 0 : 1;
}
