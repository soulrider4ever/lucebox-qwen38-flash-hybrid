#include "qwen4exp_expert_shards.h"

#include "qwen4exp_gguf_inventory.h"

#include <algorithm>
#include <limits>
#include <memory>

namespace dflash::common {

namespace {

struct GgufDeleter {
    void operator()(gguf_context * ctx) const { if (ctx) gguf_free(ctx); }
};

void fail(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool family_for(const std::string & name, Qwen4ExpExpertFamily & family) {
    if (name.find("ffn_gate_exps") != std::string::npos) {
        family = Qwen4ExpExpertFamily::Gate;
        return true;
    }
    if (name.find("ffn_up_exps") != std::string::npos) {
        family = Qwen4ExpExpertFamily::Up;
        return true;
    }
    if (name.find("ffn_down_exps") != std::string::npos) {
        family = Qwen4ExpExpertFamily::Down;
        return true;
    }
    return false;
}

} // namespace

bool Qwen4ExpExpertRegion::valid() const {
    return layer >= 0 && shard >= 0 && tensor_bytes > 0 &&
           per_expert_bytes > 0 && n_expert > 0 &&
           per_expert_bytes * static_cast<uint64_t>(n_expert) == tensor_bytes;
}

bool Qwen4ExpExpertRegion::slice(int expert, uint64_t & offset, uint64_t & bytes,
                                 std::string * error) const {
    if (!valid()) {
        fail(error, "invalid Qwen4Exp expert region");
        return false;
    }
    if (expert < 0 || expert >= n_expert) {
        fail(error, "Qwen4Exp expert id is out of range");
        return false;
    }
    if (file_offset > std::numeric_limits<uint64_t>::max() - tensor_bytes) {
        fail(error, "Qwen4Exp expert region offset overflows");
        return false;
    }
    offset = file_offset + per_expert_bytes * static_cast<uint64_t>(expert);
    bytes = per_expert_bytes;
    return true;
}

const Qwen4ExpExpertRegion * Qwen4ExpExpertShardMap::find(
        int layer, Qwen4ExpExpertFamily family) const {
    for (const auto & region : regions) {
        if (region.layer == layer && region.family == family) return &region;
    }
    return nullptr;
}

bool Qwen4ExpExpertShardMap::valid(std::string * error) const {
    if (shard_paths.empty() || shard_paths.size() != shard_sizes.size() ||
        n_layer <= 0 || n_expert <= 0) {
        fail(error, "invalid Qwen4Exp expert shard map dimensions");
        return false;
    }
    for (int layer = 0; layer < n_layer; ++layer) {
        for (const auto family : {Qwen4ExpExpertFamily::Gate,
                                  Qwen4ExpExpertFamily::Up,
                                  Qwen4ExpExpertFamily::Down}) {
            const auto * region = find(layer, family);
            if (!region || !region->valid()) {
                fail(error, "missing or invalid Qwen4Exp expert region at layer " +
                    std::to_string(layer));
                return false;
            }
            if (region->shard >= static_cast<int>(shard_sizes.size()) ||
                region->file_offset > shard_sizes[static_cast<size_t>(region->shard)] ||
                region->tensor_bytes > shard_sizes[static_cast<size_t>(region->shard)] -
                    region->file_offset) {
                fail(error, "Qwen4Exp expert region exceeds its shard");
                return false;
            }
        }
    }
    return true;
}

bool map_qwen4exp_expert_shards(const std::vector<std::string> & shard_paths,
                                Qwen4ExpExpertShardMap & out,
                                std::string * error) {
    out = {};
    if (shard_paths.empty()) {
        fail(error, "no Qwen4Exp GGUF shards supplied");
        return false;
    }

    gguf_init_params params{};
    params.no_alloc = true;
    std::vector<std::unique_ptr<gguf_context, GgufDeleter>> contexts;
    contexts.reserve(shard_paths.size());
    out.shard_paths = shard_paths;
    out.shard_sizes.resize(shard_paths.size(), 0);

    Qwen4ExpGgufInventory inventory;
    for (size_t shard_index = 0; shard_index < shard_paths.size(); ++shard_index) {
        if (shard_paths[shard_index].empty()) {
            fail(error, "empty Qwen4Exp GGUF shard path");
            return false;
        }
        std::unique_ptr<gguf_context, GgufDeleter> ctx(
            gguf_init_from_file(shard_paths[shard_index].c_str(), params));
        if (!ctx) {
            fail(error, "failed to open Qwen4Exp GGUF shard: " + shard_paths[shard_index]);
            return false;
        }
        contexts.push_back(std::move(ctx));
        // GGUF tensor offsets are relative to this data section. The caller
        // needs absolute file offsets for mmap/pread materialization.
        out.shard_sizes[shard_index] = gguf_get_data_offset(contexts.back().get());

        const int64_t count = gguf_get_n_tensors(contexts.back().get());
        for (int64_t tid = 0; tid < count; ++tid) {
            const char * raw_name = gguf_get_tensor_name(contexts.back().get(), tid);
            if (!raw_name) { fail(error, "GGUF tensor has no name"); return false; }
            const std::string name(raw_name);
            const auto identity = identify_qwen4exp_tensor(name);
            if (identity.role != Qwen4ExpTensorRole::RoutedExpert || identity.layer < 0) continue;
            Qwen4ExpExpertFamily family;
            if (!family_for(name, family)) {
                fail(error, "unknown Qwen4Exp routed expert family: " + name);
                return false;
            }
            const uint64_t bytes = static_cast<uint64_t>(
                gguf_get_tensor_size(contexts.back().get(), tid));
            Qwen4ExpExpertRegion region;
            region.layer = identity.layer;
            region.family = family;
            region.shard = static_cast<int>(shard_index);
            region.file_offset = static_cast<uint64_t>(gguf_get_data_offset(contexts.back().get())) +
                static_cast<uint64_t>(gguf_get_tensor_offset(contexts.back().get(), tid));
            region.tensor_bytes = bytes;
            out.regions.push_back(region);
        }
    }

    // Re-scan through the already-open contexts for authoritative dimensions.
    std::vector<const gguf_context *> views;
    views.reserve(contexts.size());
    for (const auto & ctx : contexts) views.push_back(ctx.get());
    if (!scan_qwen4exp_gguf_shards(views, inventory, error)) return false;
    out.n_layer = inventory.n_layer;
    out.n_expert = inventory.n_expert;
    for (auto & region : out.regions) {
        if (region.layer >= out.n_layer || region.tensor_bytes == 0 ||
            region.tensor_bytes % static_cast<uint64_t>(out.n_expert) != 0) {
            fail(error, "Qwen4Exp expert tensor has invalid layer or size");
            return false;
        }
        region.n_expert = out.n_expert;
        region.per_expert_bytes = region.tensor_bytes / static_cast<uint64_t>(out.n_expert);
    }
    if (out.regions.size() != static_cast<size_t>(out.n_layer) * 3) {
        fail(error, "Qwen4Exp expert map does not contain exactly gate/up/down per layer");
        return false;
    }
    // gguf_get_data_offset is not the file size; the map only has offsets so
    // retain a conservative end bound from the largest referenced range. A
    // materializer must still check the actual mmap length before copying.
    for (size_t i = 0; i < out.shard_sizes.size(); ++i) {
        uint64_t end = out.shard_sizes[i];
        for (const auto & region : out.regions) if (region.shard == static_cast<int>(i)) {
            end = std::max(end, region.file_offset + region.tensor_bytes);
        }
        out.shard_sizes[i] = end;
    }
    return out.valid(error);
}

} // namespace dflash::common
