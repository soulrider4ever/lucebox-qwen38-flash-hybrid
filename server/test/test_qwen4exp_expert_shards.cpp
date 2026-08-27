#include "common/qwen4exp_expert_shards.h"

#include <cassert>
#include <cstdint>
#include <string>

using namespace dflash::common;

int main() {
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
    return 0;
}
