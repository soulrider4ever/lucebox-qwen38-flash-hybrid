#include "qwen4exp_gguf_inventory.h"

#include <algorithm>
#include <cctype>
#include <limits>

namespace dflash::common {

namespace {

bool read_u32(const gguf_context * g, const char * key, int & out) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    const uint32_t value = gguf_get_val_u32(g, id);
    if (value > static_cast<uint32_t>(std::numeric_limits<int>::max())) return false;
    out = static_cast<int>(value);
    return true;
}

bool read_arch(const gguf_context * g, std::string & out) {
    const int64_t id = gguf_find_key(g, "general.architecture");
    if (id < 0) return false;
    const char * value = gguf_get_val_str(g, id);
    if (!value || !*value) return false;
    out = value;
    return true;
}

bool parse_layer(const Qwen4ExpTensorIdentity & identity, int & layer) {
    if (!identity.valid() || identity.layer < 0) return false;
    layer = identity.layer;
    return true;
}

void fail(std::string * error, const std::string & message) {
    if (error) *error = message;
}

} // namespace

bool Qwen4ExpGgufInventory::valid(std::string * error) const {
    if (architecture != "qwen4exp") {
        fail(error, "GGUF architecture is not qwen4exp: " + architecture);
        return false;
    }
    if (n_layer <= 0 || n_expert <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_expert || static_cast<int>(layers.size()) != n_layer) {
        fail(error, "invalid Qwen4Exp GGUF dimensions");
        return false;
    }
    if (first_routed_layer < 0 || first_routed_layer >= n_layer) {
        fail(error, "Qwen4Exp GGUF has no routed expert layer");
        return false;
    }
    for (int layer = first_routed_layer; layer < n_layer; ++layer) {
        const auto & expert = layers[static_cast<size_t>(layer)];
        if (expert.expert_tensor_count != 3 ||
            expert.gate_bytes == 0 || expert.up_bytes == 0 || expert.down_bytes == 0) {
            fail(error, "incomplete routed expert surface at Qwen4Exp layer " +
                 std::to_string(layer));
            return false;
        }
        if (expert.per_expert_bytes == 0 || expert.gate_per_expert_bytes == 0 ||
            expert.up_per_expert_bytes == 0 || expert.down_per_expert_bytes == 0) {
            fail(error, "routed expert bytes are not divisible into complete experts at Qwen4Exp layer " +
                 std::to_string(layer));
            return false;
        }
        const uint64_t experts = static_cast<uint64_t>(n_expert);
        if (expert.gate_per_expert_bytes > std::numeric_limits<uint64_t>::max() / experts ||
            expert.up_per_expert_bytes > std::numeric_limits<uint64_t>::max() / experts ||
            expert.down_per_expert_bytes > std::numeric_limits<uint64_t>::max() / experts ||
            expert.per_expert_bytes > std::numeric_limits<uint64_t>::max() / experts ||
            expert.gate_per_expert_bytes * experts != expert.gate_bytes ||
            expert.up_per_expert_bytes * experts != expert.up_bytes ||
            expert.down_per_expert_bytes * experts != expert.down_bytes ||
            expert.per_expert_bytes * experts != expert.total_bytes ||
            expert.per_expert_bytes != expert.gate_per_expert_bytes +
                expert.up_per_expert_bytes + expert.down_per_expert_bytes) {
            fail(error, "inconsistent per-expert routed byte accounting at Qwen4Exp layer " +
                 std::to_string(layer));
            return false;
        }
    }
    return true;
}

bool scan_qwen4exp_gguf_inventory(const gguf_context * gguf,
                                  Qwen4ExpGgufInventory & out,
                                  std::string * error) {
    if (!gguf) {
        fail(error, "null GGUF context");
        return false;
    }
    return scan_qwen4exp_gguf_shards({gguf}, out, error);
}

bool scan_qwen4exp_gguf_shards(const std::vector<const gguf_context *> & shards,
                               Qwen4ExpGgufInventory & out,
                               std::string * error) {
    out = {};
    if (shards.empty()) {
        fail(error, "no GGUF shards supplied");
        return false;
    }

    bool initialized = false;
    for (const gguf_context * gguf : shards) {
        if (!gguf) {
            fail(error, "null GGUF shard context");
            return false;
        }
        Qwen4ExpGgufInventory shard;
        // Read the repeated model metadata from each shard, but scan tensor
        // records into a partial inventory because split files partition the
        // expert tensors.
        const bool has_arch = read_arch(gguf, shard.architecture);
        const bool has_dimensions =
            read_u32(gguf, "qwen4exp.block_count", shard.n_layer) &&
            read_u32(gguf, "qwen4exp.expert_count", shard.n_expert) &&
            read_u32(gguf, "qwen4exp.expert_used_count", shard.n_expert_used);
        // GGUF split format stores the full model metadata only in shard 0;
        // later shards may contain tensor records and split bookkeeping only.
        if (!initialized && (!has_arch || !has_dimensions)) {
            fail(error, "first GGUF shard lacks Qwen4Exp model metadata");
            return false;
        }
        if (!initialized) {
            out.architecture = shard.architecture;
            out.n_layer = shard.n_layer;
            out.n_expert = shard.n_expert;
            out.n_expert_used = shard.n_expert_used;
            out.layers.resize(static_cast<size_t>(out.n_layer));
            initialized = true;
        } else if (has_arch && has_dimensions &&
                   (out.architecture != shard.architecture || out.n_layer != shard.n_layer ||
                   out.n_expert != shard.n_expert || out.n_expert_used != shard.n_expert_used)) {
            fail(error, "inconsistent Qwen4Exp metadata across GGUF shards");
            return false;
        }

        const int64_t tensor_count = gguf_get_n_tensors(gguf);
        for (int64_t tid = 0; tid < tensor_count; ++tid) {
            const char * raw_name = gguf_get_tensor_name(gguf, tid);
            if (!raw_name) {
                fail(error, "GGUF tensor has no name");
                return false;
            }
            const std::string name(raw_name);
            if (!out.tensor_roles.observe(name, error)) return false;

            const auto identity = identify_qwen4exp_tensor(name);
            int layer = -1;
            if (!parse_layer(identity, layer) ||
                identity.role != Qwen4ExpTensorRole::RoutedExpert) continue;
            if (layer >= out.n_layer) {
                fail(error, "Qwen4Exp tensor layer exceeds block count: " + name);
                return false;
            }
            auto & inventory = out.layers[static_cast<size_t>(layer)];
            const uint64_t bytes = static_cast<uint64_t>(gguf_get_tensor_size(gguf, tid));
            if (bytes == 0) {
                fail(error, "zero-sized routed expert tensor: " + name);
                return false;
            }
            ++inventory.expert_tensor_count;
            inventory.total_bytes += bytes;
            if (name.find("ffn_gate_exps") != std::string::npos) inventory.gate_bytes += bytes;
            else if (name.find("ffn_up_exps") != std::string::npos) inventory.up_bytes += bytes;
            else if (name.find("ffn_down_exps") != std::string::npos) inventory.down_bytes += bytes;
            else {
                fail(error, "unknown routed expert tensor family: " + name);
                return false;
            }
            if (bytes % static_cast<uint64_t>(out.n_expert) != 0) {
                fail(error, "routed expert tensor is not divisible by expert count: " + name);
                return false;
            }
            const uint64_t per_expert = bytes / static_cast<uint64_t>(out.n_expert);
            inventory.per_expert_bytes += per_expert;
            if (name.find("ffn_gate_exps") != std::string::npos) inventory.gate_per_expert_bytes += per_expert;
            else if (name.find("ffn_up_exps") != std::string::npos) inventory.up_per_expert_bytes += per_expert;
            else inventory.down_per_expert_bytes += per_expert;
            if (out.first_routed_layer < 0) out.first_routed_layer = layer;
            else out.first_routed_layer = std::min(out.first_routed_layer, layer);
        }
    }
    if (!out.valid(error)) return false;
    return out.tensor_roles.validate(out.n_layer, out.first_routed_layer, error);
}

} // namespace dflash::common
