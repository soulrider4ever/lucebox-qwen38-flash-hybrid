// Split-GGUF routed-expert data map for Qwen4Exp.
//
// The map is deliberately backend-neutral. It identifies the exact shard and
// byte range for each routed expert tensor so a future materializer can copy
// selected experts to the peer GPU without loading or concatenating the full
// model. It does not allocate GPU memory or execute the model.
#pragma once

#include "qwen4exp_hybrid_plan.h"

#include "gguf.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

enum class Qwen4ExpExpertFamily { Gate, Up, Down };

struct Qwen4ExpExpertRegion {
    int layer = -1;
    Qwen4ExpExpertFamily family = Qwen4ExpExpertFamily::Gate;
    int shard = -1;
    uint64_t file_offset = 0;       // absolute offset, including GGUF data section
    uint64_t tensor_bytes = 0;      // complete stacked tensor
    uint64_t per_expert_bytes = 0;  // one contiguous expert slice
    int n_expert = 0;

    bool valid() const;
    bool slice(int expert, uint64_t & offset, uint64_t & bytes,
               std::string * error = nullptr) const;
};

struct Qwen4ExpExpertShardMap {
    std::vector<std::string> shard_paths;
    std::vector<uint64_t> shard_sizes;
    std::vector<Qwen4ExpExpertRegion> regions;
    int n_layer = 0;
    int n_expert = 0;

    bool valid(std::string * error = nullptr) const;
    const Qwen4ExpExpertRegion * find(int layer, Qwen4ExpExpertFamily family) const;
};

// Read all split shards with no_alloc GGUF contexts and build a validated map.
// The returned offsets are safe to use with a separately mmap'd shard file.
bool map_qwen4exp_expert_shards(const std::vector<std::string> & shard_paths,
                                Qwen4ExpExpertShardMap & out,
                                std::string * error = nullptr);

} // namespace dflash::common
