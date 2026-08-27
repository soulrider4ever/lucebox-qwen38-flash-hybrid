#include "common/gguf_mmap.h"
#include "common/qwen4exp_expert_materializer.h"
#include "common/qwen4exp_hybrid_loader.h"
#include "common/moe_hybrid_routing_stats.h"
#include "common/moe_hybrid_ffn_eval.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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
    MoeHybridRoutingStats routing;
    const char * hotness_path = std::getenv("QWEN4EXP_HOTNESS_CSV");
    const char * budget_text = std::getenv("QWEN4EXP_HOT_BUDGET_MIB");
    if (hotness_path && hotness_path[0]) {
        if (!routing.load_csv(hotness_path, routing, &error)) {
            std::fprintf(stderr, "hotness failed: %s\n", error.c_str());
            return 1;
        }
        char * end = nullptr;
        const unsigned long long budget_mib = budget_text
            ? std::strtoull(budget_text, &end, 10) : 28672ULL;
        if ((budget_text && (!end || *end != '\0')) || budget_mib == 0) {
            std::fprintf(stderr, "QWEN4EXP_HOT_BUDGET_MIB is invalid\n");
            return 1;
        }
        const uint64_t budget = budget_mib * 1024ULL * 1024ULL;
        const char * mode = std::getenv("QWEN4EXP_PLACEMENT");
        if (mode && std::strcmp(mode, "coverage") == 0) {
            std::vector<uint64_t> layer_bytes(static_cast<size_t>(shard_map.n_layer), 0);
            for (int il = 0; il < shard_map.n_layer; ++il) {
                for (Qwen4ExpExpertFamily family : {
                        Qwen4ExpExpertFamily::Gate,
                        Qwen4ExpExpertFamily::Up,
                        Qwen4ExpExpertFamily::Down}) {
                    const auto * region = shard_map.find(il, family);
                    if (!region) {
                        std::fprintf(stderr, "missing layer %d expert region\n", il);
                        return 1;
                    }
                    layer_bytes[static_cast<size_t>(il)] += region->per_expert_bytes;
                }
            }
            if (!MoeHybridPlacement::build_from_stats_with_layer_bytes(
                    routing, layer_bytes, budget, 10, placement, &error)) {
                std::fprintf(stderr, "coverage plan failed: %s\n", error.c_str());
                return 1;
            }
        } else {
            Qwen4ExpHybridPlan plan;
            std::vector<uint64_t> fixed(static_cast<size_t>(shard_map.n_layer), 0);
            if (!prepare_qwen4exp_hybrid_plan_from_gguf(
                    shards, routing, fixed, budget, 1.0, plan, &error)) {
                std::fprintf(stderr, "plan failed: %s\n", error.c_str());
                return 1;
            }
            placement = std::move(plan.experts);
        }
    } else {
        placement.n_layer = shard_map.n_layer;
        placement.n_expert = shard_map.n_expert;
        placement.n_expert_used = 10;
        placement.total_hot = shard_map.n_layer;
        placement.hot_counts.assign(static_cast<size_t>(shard_map.n_layer), 1);
        placement.hot_expert_ids.assign(static_cast<size_t>(shard_map.n_layer), {0});
    }
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

    BackendGuard peer;
    const bool materialize_peer = std::getenv("QWEN4EXP_MATERIALIZE_PEER") != nullptr;
    if (materialize_peer) {
        peer.backend = ggml_backend_cuda_init(1);
        if (!peer.backend) {
            std::fprintf(stderr, "failed to initialize HIP device 1\n");
            return 1;
        }
    }

    MoeHybridStorage storage;
    Qwen4ExpExpertMaterializeConfig config;
    config.primary_backend = primary.backend;
    config.peer_backend = peer.backend;
    config.materialize_peer = materialize_peer;
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
    const int probe_expert = placement.hot_expert_ids[0][0];
    const size_t probe = static_cast<size_t>(
        std::min<uint64_t>(gate->per_expert_bytes, 64 * 1024));
    std::vector<uint8_t> uploaded(probe);
    ggml_backend_tensor_get(storage.layers[0].gate_hot,
                            uploaded.data(), 0, uploaded.size());
    const auto * expected = static_cast<const uint8_t *>(source.data()) +
        gate->file_offset + (uint64_t) probe_expert * gate->per_expert_bytes;
    if (std::memcmp(uploaded.data(), expected, uploaded.size()) != 0) {
        std::fprintf(stderr, "R9700 expert upload does not match GGUF bytes\n");
        return 1;
    }

    size_t allocated = 0;
    for (const MoeHybridLayerStorage & layer : storage.layers) {
        if (layer.hot_buf) allocated += ggml_backend_buffer_get_size(layer.hot_buf);
    }
    double coverage = 0.0;
    if (routing.matches(shard_map.n_layer, shard_map.n_expert, 10)) {
        uint64_t selected = 0;
        uint64_t total = 0;
        for (int il = 0; il < shard_map.n_layer; ++il) {
            total += routing.layer_totals[static_cast<size_t>(il)];
            for (int32_t expert : placement.hot_expert_ids[static_cast<size_t>(il)]) {
                selected += routing.counts[(size_t) il * shard_map.n_expert + (size_t) expert];
            }
        }
        coverage = total ? 100.0 * (double) selected / (double) total : 0.0;
    }
    std::printf("PASS device=0 layers=%d hot_experts=%d bytes=%zu coverage=%.3f%% probe_expert=%d probe=%zu\n",
                shard_map.n_layer, placement.total_hot, allocated, coverage,
                probe_expert, probe);

    if (materialize_peer && std::getenv("QWEN4EXP_MICROBENCH")) {
        const auto & layer = storage.layers[0];
        if (!layer.gate_hot || !layer.down_hot || !layer.gate_cold || !layer.down_cold) {
            std::fprintf(stderr, "dual-owner materialization is incomplete\n");
            return 1;
        }
        MoeHybridConfig cfg;
        cfg.n_layer = shard_map.n_layer;
        cfg.n_expert = shard_map.n_expert;
        cfg.n_expert_used = 10;
        cfg.n_embd = static_cast<int>(layer.gate_hot->ne[0]);
        cfg.n_ff_exp = static_cast<int>(layer.gate_hot->ne[1]);
        cfg.first_moe_layer = 0;
        cfg.cold_expert_backend = MoeHybridColdBackend::Gpu;
        cfg.materialize_hot_experts = true;
        cfg.materialize_cold_experts = true;

        ggml_init_params gp{};
        gp.mem_size = 128 * 1024 * 1024;
        gp.no_alloc = true;
        ggml_context * ctx = ggml_init(gp);
        if (!ctx) return 1;
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_embd, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cfg.n_expert_used, 1);
        ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_expert_used, 1);
        ggml_set_input(input);
        ggml_set_input(ids);
        ggml_set_input(weights);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        MoeLayerDesc desc;
        MoeHybridGraphInputs graph_inputs;
        if (!build_moe_hybrid_ffn_graph(
                ctx, graph, cfg, desc, layer, input, ids, weights, 1,
                graph_inputs, false, false,
                MoeHybridJoinMode::OwnerPartialSums,
                MoeHybridRouteBalance::Disabled) || !graph_inputs.output) {
            std::fprintf(stderr, "failed to build dual-owner FFN graph\n");
            ggml_free(ctx);
            return 1;
        }
        ggml_set_output(graph_inputs.output);
        ggml_build_forward_expand(graph, graph_inputs.output);
        BackendGuard cpu{ggml_backend_cpu_init()};
        if (!cpu.backend) {
            ggml_free(ctx);
            return 1;
        }
        ggml_backend_t backends[] = {primary.backend, peer.backend, cpu.backend};
        ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, 3, 256, false, true);
        if (!sched) {
            ggml_free(ctx);
            return 1;
        }
        auto bind = [&](const std::vector<ggml_tensor *> & nodes, ggml_backend_t backend) {
            for (ggml_tensor * node : nodes) {
                if (node) ggml_backend_sched_set_tensor_backend(sched, node, backend);
            }
        };
        ggml_backend_sched_set_tensor_backend(sched, input, primary.backend);
        ggml_backend_sched_set_tensor_backend(sched, ids, primary.backend);
        ggml_backend_sched_set_tensor_backend(sched, weights, primary.backend);
        bind(graph_inputs.hot_remap_nodes, primary.backend);
        bind(graph_inputs.hot_nodes, primary.backend);
        bind(graph_inputs.cold_remap_nodes, peer.backend);
        bind(graph_inputs.cold_nodes, peer.backend);
        bind(graph_inputs.join_nodes, primary.backend);
        if (graph_inputs.hot_local_lut) ggml_backend_sched_set_tensor_backend(sched, graph_inputs.hot_local_lut, primary.backend);
        if (graph_inputs.hot_valid_lut) ggml_backend_sched_set_tensor_backend(sched, graph_inputs.hot_valid_lut, primary.backend);
        if (graph_inputs.cold_local_lut) ggml_backend_sched_set_tensor_backend(sched, graph_inputs.cold_local_lut, peer.backend);
        if (graph_inputs.cold_valid_lut) ggml_backend_sched_set_tensor_backend(sched, graph_inputs.cold_valid_lut, peer.backend);
        ggml_backend_sched_set_tensor_backend(sched, graph_inputs.output, primary.backend);
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            std::fprintf(stderr, "failed to allocate dual-owner FFN graph\n");
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            return 1;
        }

        std::vector<float> host_input(static_cast<size_t>(cfg.n_embd), 0.01f);
        std::vector<int32_t> host_ids;
        host_ids.reserve(cfg.n_expert_used);
        int requested_hot = 9;
        if (const char * text = std::getenv("QWEN4EXP_MICROBENCH_HOT_ROUTES")) {
            char * end = nullptr;
            const long parsed = std::strtol(text, &end, 10);
            if (end != text && *end == '\0' && parsed >= 0 && parsed <= cfg.n_expert_used) {
                requested_hot = static_cast<int>(parsed);
            }
        }
        const int hot_take = std::min<int>(requested_hot, layer.hot_expert_ids.size());
        for (int i = 0; i < hot_take; ++i) host_ids.push_back(layer.hot_expert_ids[(size_t)i]);
        for (int i = 0; (int)host_ids.size() < cfg.n_expert_used && i < (int)layer.cold_expert_ids.size(); ++i) {
            host_ids.push_back(layer.cold_expert_ids[(size_t)i]);
        }
        if ((int)host_ids.size() != cfg.n_expert_used) {
            std::fprintf(stderr, "not enough dual-owner routes for microbench\n");
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            return 1;
        }
        std::vector<float> host_weights(static_cast<size_t>(cfg.n_expert_used), 0.1f);
        ggml_backend_tensor_set(input, host_input.data(), 0, ggml_nbytes(input));
        ggml_backend_tensor_set(ids, host_ids.data(), 0, ggml_nbytes(ids));
        ggml_backend_tensor_set(weights, host_weights.data(), 0, ggml_nbytes(weights));
        std::vector<double> milliseconds;
        for (int iteration = 0; iteration < 13; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
            ggml_backend_synchronize(primary.backend);
            ggml_backend_synchronize(peer.backend);
            const auto end = std::chrono::steady_clock::now();
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "dual-owner FFN compute failed\n");
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                return 1;
            }
            if (iteration >= 3) {
                milliseconds.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
        std::vector<float> result(static_cast<size_t>(cfg.n_embd));
        ggml_backend_tensor_get(graph_inputs.output, result.data(), 0, ggml_nbytes(graph_inputs.output));
        if (!std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); })) {
            std::fprintf(stderr, "dual-owner FFN output is non-finite\n");
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            return 1;
        }
        std::sort(milliseconds.begin(), milliseconds.end());
        std::printf("MICROBENCH layer=0 routes_hot=%d routes_cold=%d median_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
                    hot_take, cfg.n_expert_used - hot_take,
                    milliseconds[milliseconds.size()/2], milliseconds.front(), milliseconds.back());
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    return 0;
}
