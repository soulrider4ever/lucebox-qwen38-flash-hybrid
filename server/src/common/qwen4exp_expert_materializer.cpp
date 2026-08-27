#include "qwen4exp_expert_materializer.h"

#include "gguf_mmap.h"
#include "qwen4exp_gguf_inventory.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace dflash::common {

namespace {

struct GgufView {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;

    GgufView() = default;
    GgufView(const GgufView &) = delete;
    GgufView & operator=(const GgufView &) = delete;
    GgufView(GgufView && other) noexcept
        : gguf(other.gguf), meta(other.meta) {
        other.gguf = nullptr;
        other.meta = nullptr;
    }
    GgufView & operator=(GgufView && other) noexcept {
        if (this != &other) {
            reset();
            gguf = other.gguf;
            meta = other.meta;
            other.gguf = nullptr;
            other.meta = nullptr;
        }
        return *this;
    }
    ~GgufView() { reset(); }
    void reset() {
        if (gguf) gguf_free(gguf);
        if (meta) ggml_free(meta);
        gguf = nullptr;
        meta = nullptr;
    }
};

void fail(std::string * error, const std::string & message) {
    if (error) *error = message;
}

bool family_for_name(const std::string & name,
                     Qwen4ExpExpertFamily & family) {
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

ggml_tensor * find_expert_tensor(const std::vector<GgufView> & views,
                                 int layer,
                                 Qwen4ExpExpertFamily family) {
    for (const GgufView & view : views) {
        const int64_t count = gguf_get_n_tensors(view.gguf);
        for (int64_t tid = 0; tid < count; ++tid) {
            const char * raw = gguf_get_tensor_name(view.gguf, tid);
            if (!raw) continue;
            const std::string name(raw);
            const Qwen4ExpTensorIdentity identity = identify_qwen4exp_tensor(name);
            if (identity.role != Qwen4ExpTensorRole::RoutedExpert ||
                identity.layer != layer) continue;
            Qwen4ExpExpertFamily found;
            if (!family_for_name(name, found) || found != family) continue;
            return ggml_get_tensor(view.meta, raw);
        }
    }
    return nullptr;
}

bool bounded_region(const Qwen4ExpExpertRegion & region,
                    const std::vector<GgufMmap> & maps,
                    ExpertTensorFileData & out,
                    std::string * error) {
    if (!region.valid() || region.shard < 0 ||
        region.shard >= static_cast<int>(maps.size())) {
        fail(error, "invalid Qwen4Exp expert region during materialization");
        return false;
    }
    const GgufMmap & map = maps[static_cast<size_t>(region.shard)];
    if (region.file_offset > map.size() ||
        region.tensor_bytes > map.size() - region.file_offset) {
        fail(error, "Qwen4Exp expert region exceeds mapped shard");
        return false;
    }
    out.data = static_cast<const uint8_t *>(map.data()) + region.file_offset;
    out.size = static_cast<size_t>(region.tensor_bytes);
    return true;
}

} // namespace

bool materialize_qwen4exp_experts_from_shards(
    const std::vector<std::string> & shard_paths,
    const MoeHybridPlacement & placement,
    const Qwen4ExpExpertMaterializeConfig & config,
    MoeHybridStorage & out,
    std::string * error) {
    if (!config.primary_backend) {
        fail(error, "Qwen4Exp primary expert backend is null");
        return false;
    }
    if (config.materialize_peer && !config.peer_backend) {
        fail(error, "Qwen4Exp peer materialization requested without a backend");
        return false;
    }

    Qwen4ExpExpertShardMap shard_map;
    if (!map_qwen4exp_expert_shards(shard_paths, shard_map, error)) return false;
    if (!placement.matches(shard_map.n_layer, shard_map.n_expert,
                           placement.n_expert_used)) {
        fail(error, "Qwen4Exp placement does not match split GGUF dimensions");
        return false;
    }

    std::vector<GgufMmap> maps(shard_paths.size());
    std::vector<GgufView> views;
    views.reserve(shard_paths.size());
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        std::string mmap_error;
        if (!maps[i].open(shard_paths[i], mmap_error)) {
            fail(error, mmap_error);
            return false;
        }
        GgufView view;
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &view.meta;
        view.gguf = gguf_init_from_file(shard_paths[i].c_str(), params);
        if (!view.gguf || !view.meta) {
            fail(error, "failed to open Qwen4Exp tensor metadata: " + shard_paths[i]);
            return false;
        }
        views.push_back(std::move(view));
    }

    std::vector<MoeLayerDesc> descs(static_cast<size_t>(shard_map.n_layer));
    std::vector<LayerExpertFileData> files(static_cast<size_t>(shard_map.n_layer));
    for (int layer = 0; layer < shard_map.n_layer; ++layer) {
        MoeLayerDesc & desc = descs[static_cast<size_t>(layer)];
        desc.ffn_gate_exps = find_expert_tensor(
            views, layer, Qwen4ExpExpertFamily::Gate);
        desc.ffn_up_exps = find_expert_tensor(
            views, layer, Qwen4ExpExpertFamily::Up);
        desc.ffn_down_exps = find_expert_tensor(
            views, layer, Qwen4ExpExpertFamily::Down);
        if (!desc.ffn_gate_exps || !desc.ffn_up_exps || !desc.ffn_down_exps) {
            fail(error, "missing Qwen4Exp expert tensor metadata at layer " +
                        std::to_string(layer));
            return false;
        }

        LayerExpertFileData & file = files[static_cast<size_t>(layer)];
        if (!bounded_region(*shard_map.find(layer, Qwen4ExpExpertFamily::Gate),
                            maps, file.gate_exps, error) ||
            !bounded_region(*shard_map.find(layer, Qwen4ExpExpertFamily::Up),
                            maps, file.up_exps, error) ||
            !bounded_region(*shard_map.find(layer, Qwen4ExpExpertFamily::Down),
                            maps, file.down_exps, error)) {
            return false;
        }
    }

    MoeHybridConfig hybrid;
    hybrid.n_layer = shard_map.n_layer;
    hybrid.n_expert = shard_map.n_expert;
    hybrid.n_expert_used = placement.n_expert_used;
    hybrid.first_moe_layer = 0;
    hybrid.materialize_hot_experts = true;
    hybrid.materialize_cold_experts = config.materialize_peer;
    hybrid.cold_expert_backend = config.materialize_peer
        ? MoeHybridColdBackend::Gpu : MoeHybridColdBackend::Cpu;

    return build_moe_hybrid_storage_from_file(
        hybrid, config.primary_backend, placement, descs, files, out, error,
        0, config.materialize_peer, config.peer_backend);
}

} // namespace dflash::common
