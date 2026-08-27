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
    }
    return true;
}

bool scan_qwen4exp_gguf_inventory(const gguf_context * gguf,
                                  Qwen4ExpGgufInventory & out,
                                  std::string * error) {
    out = {};
    if (!gguf) {
        fail(error, "null GGUF context");
        return false;
    }
    if (!read_arch(gguf, out.architecture)) {
        fail(error, "missing general.architecture in GGUF");
        return false;
    }
    if (!read_u32(gguf, "qwen4exp.block_count", out.n_layer) ||
        !read_u32(gguf, "qwen4exp.expert_count", out.n_expert) ||
        !read_u32(gguf, "qwen4exp.expert_used_count", out.n_expert_used)) {
        fail(error, "missing Qwen4Exp expert metadata in GGUF");
        return false;
    }
    out.layers.resize(static_cast<size_t>(out.n_layer));

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
        if (out.first_routed_layer < 0) out.first_routed_layer = layer;
        else out.first_routed_layer = std::min(out.first_routed_layer, layer);
    }
    if (!out.valid(error)) return false;
    return out.tensor_roles.validate(out.n_layer, out.first_routed_layer, error);
}

} // namespace dflash::common
