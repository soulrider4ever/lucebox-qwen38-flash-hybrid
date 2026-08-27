#include "qwen4exp_internal.h"

#include "common/gguf_mmap.h"
#include "common/qwen4exp_gguf_inventory.h"

#include "gguf.h"

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>

namespace dflash::common {

namespace {

struct ShardView {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    GgufMmap mmap;
    ShardView() = default;
    ShardView(const ShardView &) = delete;
    ShardView & operator=(const ShardView &) = delete;
    ShardView(ShardView && other) noexcept
        : gguf(other.gguf), meta(other.meta), mmap(std::move(other.mmap)) {
        other.gguf = nullptr;
        other.meta = nullptr;
    }
    ~ShardView() {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
    }
};

struct PendingTensor {
    std::string name;
    Qwen4ExpTensorIdentity identity;
    int shard = -1;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    ggml_tensor * source = nullptr;
};

void fail(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool is_ple_gather_table(const std::string & name) {
    return name == "per_layer_token_embd.weight" ||
           name == "per_layer_tok_embd.weight";
}

bool required_core_surface(const Qwen4ExpCoreWeights & weights,
                           std::string * error) {
    const char * global[] = {
        "token_embd.weight", "output_hc_norm.weight",
        "output_hc_down.weight", "output_hc_up.weight",
    };
    for (const char * name : global) {
        if (!weights.find(name)) {
            fail(error, std::string("missing Qwen4Exp core tensor: ") + name);
            return false;
        }
    }
    for (int layer = 0; layer < weights.n_layer; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const char * leaves[] = {
            "hc_attn_norm.weight", "hc_attn_down.weight",
            "hc_attn_up.weight", "hc_attn_inject.weight",
            "hc_ffn_norm.weight", "hc_ffn_down.weight",
            "hc_ffn_up.weight", "hc_ffn_inject.weight",
            "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        };
        for (const char * leaf : leaves) {
            if (!weights.find(prefix + leaf)) {
                fail(error, "missing Qwen4Exp layer core tensor: " + prefix + leaf);
                return false;
            }
        }
    }
    return true;
}

} // namespace

void free_qwen4exp_core_weights(Qwen4ExpCoreWeights & weights) {
    if (weights.buffer) ggml_backend_buffer_free(weights.buffer);
    if (weights.ctx) ggml_free(weights.ctx);
    weights = {};
}

bool load_qwen4exp_protected_core_from_shards(
    const std::vector<std::string> & shard_paths,
    ggml_backend_t backend,
    Qwen4ExpCoreWeights & out,
    std::string * error) {
    free_qwen4exp_core_weights(out);
    if (!backend || shard_paths.empty()) {
        fail(error, "Qwen4Exp core loader requires a backend and GGUF shards");
        return false;
    }

    std::vector<ShardView> shards;
    shards.reserve(shard_paths.size());
    std::vector<const gguf_context *> gguf_views;
    std::vector<PendingTensor> pending;
    std::unordered_set<std::string> names;
    uint64_t expert_bytes = 0;
    uint64_t ple_bytes = 0;

    for (size_t shard_index = 0; shard_index < shard_paths.size(); ++shard_index) {
        ShardView shard;
        std::string mmap_error;
        if (!shard.mmap.open(shard_paths[shard_index], mmap_error)) {
            fail(error, mmap_error);
            return false;
        }
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &shard.meta;
        shard.gguf = gguf_init_from_file(shard_paths[shard_index].c_str(), params);
        if (!shard.gguf || !shard.meta) {
            fail(error, "failed to open Qwen4Exp shard metadata: " +
                        shard_paths[shard_index]);
            return false;
        }
        const uint64_t data_offset = gguf_get_data_offset(shard.gguf);
        const int64_t count = gguf_get_n_tensors(shard.gguf);
        for (int64_t tid = 0; tid < count; ++tid) {
            const char * raw = gguf_get_tensor_name(shard.gguf, tid);
            if (!raw || !names.insert(raw).second) {
                fail(error, raw ? "duplicate Qwen4Exp tensor across shards: " +
                    std::string(raw) : "unnamed Qwen4Exp tensor");
                return false;
            }
            const uint64_t bytes = gguf_get_tensor_size(shard.gguf, tid);
            const uint64_t offset = data_offset + gguf_get_tensor_offset(shard.gguf, tid);
            if (offset > shard.mmap.size() || bytes > shard.mmap.size() - offset) {
                fail(error, "Qwen4Exp tensor exceeds mapped shard: " + std::string(raw));
                return false;
            }
            const Qwen4ExpTensorIdentity identity = identify_qwen4exp_tensor(raw);
            if (identity.role == Qwen4ExpTensorRole::RoutedExpert) {
                expert_bytes += bytes;
                continue;
            }
            if (is_ple_gather_table(raw)) {
                ple_bytes += bytes;
                continue;
            }
            ggml_tensor * source = ggml_get_tensor(shard.meta, raw);
            if (!source) {
                fail(error, "Qwen4Exp metadata tensor is missing: " + std::string(raw));
                return false;
            }
            pending.push_back({raw, identity, static_cast<int>(shard_index),
                               offset, bytes, source});
        }
        shards.push_back(std::move(shard));
        gguf_views.push_back(shards.back().gguf);
    }

    Qwen4ExpGgufInventory inventory;
    if (!scan_qwen4exp_gguf_shards(gguf_views, inventory, error)) return false;

    ggml_init_params init{};
    init.mem_size = (pending.size() + 16) * ggml_tensor_overhead() + 64 * 1024;
    init.no_alloc = true;
    out.ctx = ggml_init(init);
    if (!out.ctx) {
        fail(error, "failed to allocate Qwen4Exp core metadata context");
        return false;
    }
    out.backend = backend;
    out.n_layer = inventory.n_layer;
    out.routed_expert_bytes = expert_bytes;
    out.ple_table_bytes = ple_bytes;
    out.tensors.reserve(pending.size());
    out.by_name.reserve(pending.size());

    for (const PendingTensor & item : pending) {
        const int dims = ggml_n_dims(item.source);
        ggml_tensor * tensor = ggml_new_tensor(
            out.ctx, item.source->type, dims, item.source->ne);
        ggml_set_name(tensor, item.name.c_str());
        out.tensors.push_back({item.name, item.identity.role,
                               item.identity.layer, item.shard,
                               item.offset, item.bytes, tensor});
        out.by_name.emplace(item.name, tensor);
        out.resident_bytes += item.bytes;
    }

    if (!required_core_surface(out, error)) {
        free_qwen4exp_core_weights(out);
        return false;
    }
    out.buffer = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buffer) {
        fail(error, "failed to allocate Qwen4Exp protected core buffer");
        free_qwen4exp_core_weights(out);
        return false;
    }
    for (const Qwen4ExpCoreTensor & item : out.tensors) {
        const ShardView & shard = shards[static_cast<size_t>(item.shard)];
        const auto * source = static_cast<const uint8_t *>(shard.mmap.data()) +
            item.file_offset;
        ggml_backend_tensor_set(item.tensor, source, 0,
                                static_cast<size_t>(item.file_bytes));
    }
    return true;
}

} // namespace dflash::common
