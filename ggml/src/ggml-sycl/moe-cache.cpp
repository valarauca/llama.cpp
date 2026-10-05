//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "moe-cache.hpp"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static constexpr size_t MOE_CACHE_ALIGNMENT  = 256;
static constexpr size_t MOE_CACHE_MAX_BUFFER = 4ull << 30;

static int ggml_sycl_moe_cache_slots() {
    static const int slots = std::max(1, ggml_sycl_get_env("GGML_SYCL_MOE_CACHE_SLOTS", 32));
    return slots;
}

// One cached expert tensor of a layer and the device slot pool holding its resident experts.
struct moe_cache_member {
    const ggml_tensor *     tensor = nullptr;
    ggml_backend_buffer_t   pool   = nullptr;
    ggml_tensor_extra_gpu * extra  = nullptr;
    bool                    used   = false;
};

// The cached expert tensors of one layer. They share routing, so they share one LRU over the slots, and the
// first of them to run in a graph plans for all of them.
struct moe_cache_group {
    int                           layer    = -1;
    int64_t                       n_expert = 0;
    int                           n_slots  = 0;
    std::vector<moe_cache_member> members;
    std::vector<int32_t>          slot_of_expert;
    std::vector<int32_t>          expert_of_slot;
    std::vector<uint64_t>         last_used;
    uint64_t                      tick     = 0;
    const void *                  plan_ids = nullptr;
    ggml_backend_buffer_t         ids_buf  = nullptr;
    int32_t *                     ids_host = nullptr;
    size_t                        ids_cap  = 0;
};

// Per-device registry of groups plus counters reported when the last cached tensor is freed.
struct moe_cache_device {
    std::vector<std::unique_ptr<moe_cache_group>> groups;
    uint64_t                                      ops       = 0;
    uint64_t                                      hits      = 0;
    uint64_t                                      misses    = 0;
    uint64_t                                      bytes_h2d = 0;
};

static std::mutex       g_moe_cache_mutex;
static moe_cache_device g_moe_cache[GGML_SYCL_MAX_DEVICES];

struct moe_cache_buffer_context {
    int                                  device;
    void *                               ptr;
    std::vector<ggml_tensor_extra_gpu *> extras;
    std::vector<const ggml_tensor *>     tensors;
};

struct moe_cache_buft_context {
    int         device;
    std::string name;
};

static sycl::queue & moe_cache_queue(int device) {
    return dpct::dev_mgr::instance().get_device(device).default_queue();
}

static bool moe_cache_reorderable(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}

static ggml_backend_buffer_t moe_cache_device_alloc(int device, size_t size) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_sycl_buffer_type(device), size);
    if (buf == nullptr) {
        GGML_ABORT("%s: failed to allocate %.2f MiB of device memory for the MoE cache, lower GGML_SYCL_MOE_CACHE_SLOTS",
                   __func__, size / 1048576.0);
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buf, 0);
    return buf;
}

static void moe_cache_register(int device, const ggml_tensor * tensor) {
    int        layer = -1;
    const bool named = sscanf(tensor->name, "blk.%d.", &layer) == 1;

    std::lock_guard<std::mutex> lock(g_moe_cache_mutex);
    moe_cache_device &          mc = g_moe_cache[device];
    moe_cache_group *           g  = nullptr;
    for (auto & it : mc.groups) {
        if (!named || it->layer != layer || it->n_expert != tensor->ne[2]) {
            continue;
        }
        bool clash = false;
        for (const moe_cache_member & m : it->members) {
            clash = clash || strcmp(m.tensor->name, tensor->name) == 0;
        }
        if (!clash) {
            g = it.get();
            break;
        }
    }
    if (g == nullptr) {
        mc.groups.emplace_back(new moe_cache_group);
        g           = mc.groups.back().get();
        g->layer    = named ? layer : -1;
        g->n_expert = tensor->ne[2];
        g->n_slots  = (int) std::min<int64_t>(ggml_sycl_moe_cache_slots(), tensor->ne[2]);
        g->slot_of_expert.assign(g->n_expert, -1);
        g->expert_of_slot.assign(g->n_slots, -1);
        g->last_used.assign(g->n_slots, 0);
    }

    moe_cache_member m;
    m.tensor = tensor;
    m.pool   = moe_cache_device_alloc(device, g->n_slots * tensor->nb[2] + ggml_row_size(tensor->type, MATRIX_ROW_PADDING));
    m.extra  = new ggml_tensor_extra_gpu{};
    g->members.push_back(m);
}

static void moe_cache_unregister(int device, const ggml_tensor * tensor) {
    std::lock_guard<std::mutex> lock(g_moe_cache_mutex);
    moe_cache_device &          mc = g_moe_cache[device];
    for (size_t gi = 0; gi < mc.groups.size(); ++gi) {
        moe_cache_group & g = *mc.groups[gi];
        for (size_t mi = 0; mi < g.members.size(); ++mi) {
            if (g.members[mi].tensor != tensor) {
                continue;
            }
            ggml_backend_buffer_free(g.members[mi].pool);
            release_extra_gpu(g.members[mi].extra);
            g.members.erase(g.members.begin() + mi);
            if (g.members.empty()) {
                if (g.ids_buf) {
                    ggml_backend_buffer_free(g.ids_buf);
                }
                if (g.ids_host) {
                    sycl::free(g.ids_host, moe_cache_queue(device));
                }
                mc.groups.erase(mc.groups.begin() + gi);
            }
            if (mc.groups.empty() && mc.ops > 0) {
                GGML_LOG_INFO("%s: SYCL%d MoE cache: %llu ops, %llu expert hits, %llu misses (%.1f%% hit), %.2f GiB copied\n",
                              __func__, device, (unsigned long long) mc.ops, (unsigned long long) mc.hits,
                              (unsigned long long) mc.misses, 100.0 * mc.hits / std::max<uint64_t>(1, mc.hits + mc.misses),
                              mc.bytes_h2d / double(1ull << 30));
                mc.ops = mc.hits = mc.misses = mc.bytes_h2d = 0;
            }
            return;
        }
    }
}

static void moe_cache_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (moe_cache_buffer_context *) buffer->context;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
    for (const ggml_tensor * t : ctx->tensors) {
        moe_cache_unregister(ctx->device, t);
    }
    for (ggml_tensor_extra_gpu * extra : ctx->extras) {
        release_extra_gpu(extra);
    }
    if (ctx->ptr) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::free(ctx->ptr, moe_cache_queue(ctx->device))));
    }
    delete ctx;
}

static void * moe_cache_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((moe_cache_buffer_context *) buffer->context)->ptr;
}

static ggml_status moe_cache_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = (moe_cache_buffer_context *) buffer->context;
    if (tensor->view_src != nullptr) {
        return GGML_STATUS_SUCCESS;
    }
    if (g_ggml_sycl_enable_optimize && moe_cache_reorderable(tensor->type)) {
        auto * extra  = new ggml_tensor_extra_gpu{};
        tensor->extra = extra;
        ctx->extras.push_back(extra);
    }
    if (tensor->ne[2] > 1 && tensor->ne[3] == 1 && ggml_nbytes(tensor) == (size_t) tensor->ne[2] * tensor->nb[2]) {
        moe_cache_register(ctx->device, tensor);
        ctx->tensors.push_back(tensor);
    }
    return GGML_STATUS_SUCCESS;
}

static void moe_cache_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                           size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void moe_cache_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                                        size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void moe_cache_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
                                        size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool moe_cache_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void moe_cache_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (moe_cache_buffer_context *) buffer->context;
    if (ctx->ptr) {
        memset(ctx->ptr, value, buffer->size);
    }
}

static const ggml_backend_buffer_i moe_cache_buffer_iface = {
    /* .free_buffer   = */ moe_cache_buffer_free,
    /* .get_base      = */ moe_cache_buffer_get_base,
    /* .init_tensor   = */ moe_cache_buffer_init_tensor,
    /* .memset_tensor = */ moe_cache_buffer_memset_tensor,
    /* .set_tensor    = */ moe_cache_buffer_set_tensor,
    /* .get_tensor    = */ moe_cache_buffer_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ moe_cache_buffer_cpy_tensor,
    /* .clear         = */ moe_cache_buffer_clear,
    /* .reset         = */ NULL,
};

static const char * moe_cache_buft_get_name(ggml_backend_buffer_type_t buft) {
    return ((moe_cache_buft_context *) buft->context)->name.c_str();
}

static ggml_backend_buffer_t moe_cache_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const int device = ((moe_cache_buft_context *) buft->context)->device;
    void *    ptr    = nullptr;
    if (size > 0) {
        try {
            ptr = sycl::malloc_host(size, moe_cache_queue(device));
        } catch (...) {
            ptr = nullptr;
        }
        if (ptr == nullptr) {
            GGML_LOG_ERROR("%s: failed to allocate %.2f MiB of pinned memory\n", __func__, size / 1048576.0);
            return nullptr;
        }
    }
    auto * ctx = new moe_cache_buffer_context{ device, ptr, {}, {} };
    return ggml_backend_buffer_init(buft, moe_cache_buffer_iface, ctx, size);
}

static size_t moe_cache_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return MOE_CACHE_ALIGNMENT;
    GGML_UNUSED(buft);
}

static size_t moe_cache_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    return MOE_CACHE_MAX_BUFFER;
    GGML_UNUSED(buft);
}

static bool moe_cache_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_sycl_moe_buffer_type(int device) {
    static std::vector<ggml_backend_buffer_type> bufts = [] {
        std::vector<ggml_backend_buffer_type> v(ggml_backend_sycl_get_device_count());
        for (size_t i = 0; i < v.size(); i++) {
            v[i] = {
                /* .iface    = */ {
                    /* .get_name         = */ moe_cache_buft_get_name,
                    /* .alloc_buffer     = */ moe_cache_buft_alloc_buffer,
                    /* .alloc_buffer_n   = */ NULL,
                    /* .get_alignment    = */ moe_cache_buft_get_alignment,
                    /* .get_max_size     = */ moe_cache_buft_get_max_size,
                    /* .get_alloc_size   = */ NULL,
                    /* .get_alloc_size_n = */ NULL,
                    /* .is_host          = */ moe_cache_buft_is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ new moe_cache_buft_context{ (int) i, GGML_SYCL_NAME + std::to_string(i) + "_MOE" },
            };
        }
        return v;
    }();
    GGML_ASSERT(device >= 0 && device < (int) bufts.size());
    return &bufts[device];
}

bool ggml_backend_buffer_is_sycl_moe(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft->iface.get_name == moe_cache_buft_get_name;
}

// Reads the layer's routing on the host and gives every routed expert a slot. The queue is drained before the
// ids are copied back: reading them right behind the routing kernels returned stale ids in about one of three
// Coder IQ4_XS decode runs, a race standalone tests of the same queue pattern did not reproduce.
static void moe_cache_plan(ggml_backend_sycl_context & ctx, moe_cache_device & mc, moe_cache_group & g,
                           const ggml_tensor * ids, ggml_sycl_moe_cache_reorder_t reorder) {
    const queue_ptr stream = ctx.stream();
    for (moe_cache_member & m : g.members) {
        reorder(&ctx, m.tensor);
        m.used = false;
    }

    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    const int64_t     n_used   = ids->ne[0];
    const int64_t     n_tokens = ids->ne[1];
    const size_t      n        = n_used * n_tokens;
    std::vector<char> ids_raw(ggml_nbytes(ids));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(ids_raw.data(), ids->data, ids_raw.size()).wait()));
    auto id_at = [&](int64_t t, int64_t u) {
        return *(const int32_t *) (ids_raw.data() + t * ids->nb[1] + u * ids->nb[0]);
    };

    std::vector<int32_t> need;
    std::vector<uint8_t> seen(g.n_expert, 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e = id_at(t, u);
            GGML_ASSERT(e >= 0 && e < g.n_expert);
            if (!seen[e]) {
                seen[e] = 1;
                need.push_back(e);
            }
        }
    }
    GGML_ASSERT((int) need.size() <= g.n_slots);

    g.tick++;
    for (const int32_t e : need) {
        if (g.slot_of_expert[e] >= 0) {
            g.last_used[g.slot_of_expert[e]] = g.tick;
            mc.hits++;
        }
    }
    for (const int32_t e : need) {
        if (g.slot_of_expert[e] >= 0) {
            continue;
        }
        int victim = -1;
        for (int s = 0; s < g.n_slots; ++s) {
            if (g.last_used[s] != g.tick && (victim < 0 || g.last_used[s] < g.last_used[victim])) {
                victim = s;
            }
        }
        GGML_ASSERT(victim >= 0);
        if (g.expert_of_slot[victim] >= 0) {
            g.slot_of_expert[g.expert_of_slot[victim]] = -1;
        }
        g.expert_of_slot[victim] = e;
        g.slot_of_expert[e]      = victim;
        g.last_used[victim]      = g.tick;
        mc.misses++;
        for (const moe_cache_member & m : g.members) {
            const size_t nb2 = m.tensor->nb[2];
            char *       dst = (char *) ggml_backend_buffer_get_base(m.pool) + victim * nb2;
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(dst, (const char *) m.tensor->data + e * nb2, nb2)));
            mc.bytes_h2d += nb2;
        }
    }

    if (g.ids_cap < n) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        if (g.ids_buf) {
            ggml_backend_buffer_free(g.ids_buf);
        }
        if (g.ids_host) {
            sycl::free(g.ids_host, *stream);
        }
        g.ids_buf  = moe_cache_device_alloc(ctx.device, n * sizeof(int32_t));
        g.ids_host = sycl::malloc_host<int32_t>(n, *stream);
        GGML_ASSERT(g.ids_host != nullptr);
        g.ids_cap = n;
    }
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            g.ids_host[t * n_used + u] = g.slot_of_expert[id_at(t, u)];
        }
    }
    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ggml_backend_buffer_get_base(g.ids_buf), g.ids_host, n * sizeof(int32_t))));
    g.plan_ids = ids->data;
}

void ggml_sycl_moe_cache_prepare(ggml_backend_sycl_context & ctx, const ggml_tensor * dst,
                                 ggml_sycl_moe_cache_reorder_t reorder, ggml_tensor & src0_view,
                                 ggml_tensor & ids_view) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * ids  = dst->src[2];

    std::lock_guard<std::mutex> lock(g_moe_cache_mutex);
    moe_cache_device &          mc = g_moe_cache[ctx.device];
    moe_cache_group *           g  = nullptr;
    size_t                      mi = 0;
    for (auto & it : mc.groups) {
        for (size_t i = 0; i < it->members.size() && g == nullptr; ++i) {
            if (it->members[i].tensor == src0) {
                g  = it.get();
                mi = i;
            }
        }
    }
    GGML_ASSERT(g != nullptr && "MUL_MAT_ID weights in a MoE cache buffer must be 3D expert tensors");

    if (g->plan_ids != ids->data || g->members[mi].used) {
        moe_cache_plan(ctx, mc, *g, ids, reorder);
    }
    moe_cache_member & m = g->members[mi];
    m.used               = true;
    mc.ops++;

    const ggml_tensor_extra_gpu * host_extra = (const ggml_tensor_extra_gpu *) src0->extra;
    const bool                    reordered  = host_extra && host_extra->optimized_feature.reorder;
    m.extra->optimized_feature.reorder       = reordered;

    src0_view           = *src0;
    src0_view.data      = ggml_backend_buffer_get_base(m.pool);
    src0_view.buffer    = m.pool;
    src0_view.ne[2]     = g->n_slots;
    src0_view.nb[3]     = src0->nb[2] * g->n_slots;
    src0_view.extra     = reordered ? m.extra : nullptr;
    src0_view.view_src  = nullptr;
    src0_view.view_offs = 0;

    ids_view           = *ids;
    ids_view.data      = ggml_backend_buffer_get_base(g->ids_buf);
    ids_view.buffer    = g->ids_buf;
    ids_view.nb[0]     = sizeof(int32_t);
    ids_view.nb[1]     = ids->ne[0] * sizeof(int32_t);
    ids_view.nb[2]     = ids_view.nb[1] * ids->ne[1];
    ids_view.nb[3]     = ids_view.nb[2] * ids->ne[2];
    ids_view.extra     = nullptr;
    ids_view.view_src  = nullptr;
    ids_view.view_offs = 0;
}
