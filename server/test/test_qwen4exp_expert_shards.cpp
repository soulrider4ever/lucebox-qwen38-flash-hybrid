#include "common/qwen4exp_expert_shards.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    Qwen4ExpExpertRegion region;
    region.layer = 7;
    region.shard = 2;
    region.tensor_bytes = 1200;
    region.n_expert = 10;
    region.per_expert_bytes = 120;
    assert(region.valid());

    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::string error;
    assert(region.slice(0, offset, bytes, &error));
    assert(offset == region.file_offset);
    assert(bytes == 120);
    assert(region.slice(9, offset, bytes, &error));
    assert(offset == region.file_offset + 1080);
    assert(!region.slice(10, offset, bytes, &error));

    Qwen4ExpExpertShardMap map;
    map.shard_paths = {"one.gguf", "two.gguf", "three.gguf"};
    map.shard_sizes = {1000, 1000, 1400};
    map.n_layer = 0;
    map.n_expert = 10;
    assert(!map.valid(&error));

    if (argc > 1) {
        Qwen4ExpExpertShardMap real_map;
        std::vector<std::string> paths(argv + 1, argv + argc);
        if (!map_qwen4exp_expert_shards(paths, real_map, &error)) {
            std::fprintf(stderr, "expert map rejected: %s\n", error.c_str());
            return 1;
        }
        if (real_map.n_layer != 48 || real_map.n_expert != 512 ||
            real_map.regions.size() != 48 * 3) {
            std::fprintf(stderr, "unexpected map dimensions: layers=%d experts=%d regions=%zu\n",
                         real_map.n_layer, real_map.n_expert, real_map.regions.size());
            return 1;
        }
        const auto * gate = real_map.find(1, Qwen4ExpExpertFamily::Gate);
        if (!gate || gate->per_expert_bytes == 0 ||
            !gate->slice(511, offset, bytes, &error) ||
            bytes != gate->per_expert_bytes) {
            std::fprintf(stderr, "invalid layer 1 gate slice: %s\n", error.c_str());
            return 1;
        }
        std::printf("layers=%d experts=%d regions=%zu layer1_gate_bytes=%llu\n",
                    real_map.n_layer, real_map.n_expert, real_map.regions.size(),
                    static_cast<unsigned long long>(bytes));
    }
    return 0;
}
