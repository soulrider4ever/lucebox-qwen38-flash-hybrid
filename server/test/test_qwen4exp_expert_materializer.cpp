#include "common/gguf_mmap.h"
#include "common/qwen4exp_expert_materializer.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {
struct BackendGuard {
    ggml_backend_t backend = nullptr;
    ~BackendGuard() { if (backend) ggml_backend_free(backend); }
};
} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_SHARD [MODEL_SHARD ...]\n", argv[0]);
        return 2;
    }

    std::vector<std::string> shards(argv + 1, argv + argc);
    Qwen4ExpExpertShardMap shard_map;
    std::string error;
    if (!map_qwen4exp_expert_shards(shards, shard_map, &error)) {
        std::fprintf(stderr, "map failed: %s\n", error.c_str());
        return 1;
    }

    MoeHybridPlacement placement;
    placement.n_layer = shard_map.n_layer;
    placement.n_expert = shard_map.n_expert;
    placement.n_expert_used = 10;
    placement.total_hot = shard_map.n_layer;
    placement.hot_counts.assign(static_cast<size_t>(shard_map.n_layer), 1);
    placement.hot_expert_ids.assign(static_cast<size_t>(shard_map.n_layer), {0});
    if (!placement.valid(&error)) {
        std::fprintf(stderr, "placement failed: %s\n", error.c_str());
        return 1;
    }

    // On Lucebox HIP device 0 is the R9700. The canary allocates only one
    // expert per layer (~120 MiB total) and never touches the live 8060S model.
    BackendGuard primary{ggml_backend_cuda_init(0)};
    if (!primary.backend) {
        std::fprintf(stderr, "failed to initialize HIP device 0\n");
        return 1;
    }

    MoeHybridStorage storage;
    Qwen4ExpExpertMaterializeConfig config;
    config.primary_backend = primary.backend;
    if (!materialize_qwen4exp_experts_from_shards(
            shards, placement, config, storage, &error)) {
        std::fprintf(stderr, "materialization failed: %s\n", error.c_str());
        return 1;
    }

    if (storage.layers.size() != static_cast<size_t>(shard_map.n_layer) ||
        !storage.layers[0].gate_hot || !storage.layers[0].up_hot ||
        !storage.layers[0].down_hot) {
        std::fprintf(stderr, "materialized storage is incomplete\n");
        return 1;
    }

    // Prove the upload copied the exact quantized bytes from the split shard,
    // not merely that a HIP allocation succeeded.
    const Qwen4ExpExpertRegion * gate =
        shard_map.find(0, Qwen4ExpExpertFamily::Gate);
    if (!gate) {
        std::fprintf(stderr, "layer 0 gate region missing\n");
        return 1;
    }
    GgufMmap source;
    if (!source.open(shards[static_cast<size_t>(gate->shard)], error)) {
        std::fprintf(stderr, "source mmap failed: %s\n", error.c_str());
        return 1;
    }
    const size_t probe = static_cast<size_t>(
        std::min<uint64_t>(gate->per_expert_bytes, 64 * 1024));
    std::vector<uint8_t> uploaded(probe);
    ggml_backend_tensor_get(storage.layers[0].gate_hot,
                            uploaded.data(), 0, uploaded.size());
    const auto * expected = static_cast<const uint8_t *>(source.data()) +
        gate->file_offset;
    if (std::memcmp(uploaded.data(), expected, uploaded.size()) != 0) {
        std::fprintf(stderr, "R9700 expert upload does not match GGUF bytes\n");
        return 1;
    }

    size_t allocated = 0;
    for (const MoeHybridLayerStorage & layer : storage.layers) {
        if (layer.hot_buf) allocated += ggml_backend_buffer_get_size(layer.hot_buf);
    }
    std::printf("PASS device=0 layers=%d experts_per_layer=1 bytes=%zu probe=%zu\n",
                shard_map.n_layer, allocated, probe);

    return 0;
}
