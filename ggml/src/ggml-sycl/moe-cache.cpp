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
static constexpr size_t MOE_CACHE_MEMBERS    = 4;
static constexpr size_t MOE_CACHE_COPY_WGS   = 256;
static constexpr size_t MOE_CACHE_COPY_WG    = 256;
static constexpr size_t MOE_CACHE_PLAN_WG    = 256;

static int ggml_sycl_moe_cache_slots() {
    static const int slots = std::max(1, ggml_sycl_get_env("GGML_SYCL_MOE_CACHE_SLOTS", 32));
    return slots;
}

static bool ggml_sycl_moe_cache_device_plan() {
    static const bool enabled = ggml_sycl_get_env("GGML_SYCL_MOE_CACHE_DEVICE_PLAN", 1) != 0;
    return enabled;
}

// One cached expert tensor of a layer and the device slot pool holding its resident experts.
struct moe_cache_member {
    const ggml_tensor *     tensor = nullptr;
    ggml_backend_buffer_t   pool   = nullptr;
    ggml_tensor_extra_gpu * extra       = nullptr;
    size_t                  scratch_off = 0;
    bool                    used        = false;
};

// The cached expert tensors of one layer. They share routing, so they share one LRU over the slots, and the
// first of them to run in a graph plans for all of them. A batch routed to more experts than there are slots is
// staged instead: its experts are packed into the device scratch and the LRU is left alone. A batch that can never
// route to more experts than there are slots is planned on the device against a device copy of the LRU (lru_buf),
// which is synced with the host vectors through the pinned lru_host whenever planning switches sides.
struct moe_cache_group {
    int                           layer    = -1;
    int64_t                       n_expert = 0;
    int                           n_slots  = 0;
    std::vector<moe_cache_member> members;
    std::vector<int32_t>          slot_of_expert;
    std::vector<int32_t>          expert_of_slot;
    std::vector<uint32_t>         last_used;
    uint32_t                      tick         = 0;
    bool                          staged       = false;
    int64_t                       n_packed     = 0;
    const void *                  plan_ids     = nullptr;
    ggml_backend_buffer_t         ids_buf      = nullptr;
    int32_t *                     ids_host     = nullptr;
    size_t                        ids_cap      = 0;
    ggml_backend_buffer_t         lru_buf      = nullptr;
    int32_t *                     lru_host     = nullptr;
    size_t                        expert_bytes = 0;
    bool                          host_dirty   = true;
    bool                          device_lru   = false;
};

// Word offsets into a group's device LRU buffer: the LRU state mirrored with the host (slot_of_expert at 0), then
// the device planner's miss count, miss list and 64-bit hit / miss counters.
struct moe_cache_lru_layout {
    size_t expert_of_slot;
    size_t last_used;
    size_t tick;
    size_t state_words;
    size_t miss_count;
    size_t miss_slot;
    size_t miss_expert;
    size_t counters;
    size_t words;
};

static moe_cache_lru_layout moe_cache_lru_layout_of(const moe_cache_group & g) {
    moe_cache_lru_layout l;
    l.expert_of_slot = g.n_expert;
    l.last_used      = l.expert_of_slot + g.n_slots;
    l.tick           = l.last_used + g.n_slots;
    l.state_words    = l.tick + 1;
    l.miss_count     = l.state_words;
    l.miss_slot      = l.miss_count + 1;
    l.miss_expert    = l.miss_slot + g.n_slots;
    l.counters       = GGML_PAD(l.miss_expert + g.n_slots, 2);
    l.words          = l.counters + 4;
    return l;
}

// Per-device registry of groups, the staging scratch they share (sized for the largest group, used one layer at
// a time on the in-order queue) and counters reported when the last cached tensor is freed.
struct moe_cache_device {
    std::vector<std::unique_ptr<moe_cache_group>> groups;
    ggml_backend_buffer_t                         scratch     = nullptr;
    size_t                                        scratch_cap = 0;
    uint64_t                                      ops         = 0;
    uint64_t                                      staged_ops  = 0;
    uint64_t                                      hits        = 0;
    uint64_t                                      misses      = 0;
    uint64_t                                      bytes_h2d   = 0;
    uint64_t                                      bytes_d2d   = 0;
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
        const moe_cache_lru_layout l = moe_cache_lru_layout_of(*g);
        g->lru_buf                   = moe_cache_device_alloc(device, l.words * sizeof(int32_t));
        g->lru_host                  = sycl::malloc_host<int32_t>(l.state_words, moe_cache_queue(device));
        GGML_ASSERT(g->lru_host != nullptr);
    }

    const size_t     pad = ggml_row_size(tensor->type, MATRIX_ROW_PADDING);
    moe_cache_member m;
    m.tensor = tensor;
    m.pool   = moe_cache_device_alloc(device, g->n_slots * tensor->nb[2] + pad);
    m.extra  = new ggml_tensor_extra_gpu{};
    g->members.push_back(m);
    g->expert_bytes += tensor->nb[2];

    size_t need = 0;
    for (moe_cache_member & it : g->members) {
        it.scratch_off = need;
        need += GGML_PAD(g->n_expert * it.tensor->nb[2] + ggml_row_size(it.tensor->type, MATRIX_ROW_PADDING),
                         MOE_CACHE_ALIGNMENT);
    }
    if (need > mc.scratch_cap) {
        if (mc.scratch) {
            ggml_backend_buffer_free(mc.scratch);
        }
        mc.scratch     = moe_cache_device_alloc(device, need);
        mc.scratch_cap = need;
    }
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
                const moe_cache_lru_layout l = moe_cache_lru_layout_of(g);
                uint64_t                   counters[2];
                SYCL_CHECK(CHECK_TRY_ERROR(
                    moe_cache_queue(device)
                        .memcpy(counters, (const int32_t *) ggml_backend_buffer_get_base(g.lru_buf) + l.counters,
                                sizeof(counters))
                        .wait()));
                mc.hits += counters[0];
                mc.misses += counters[1];
                mc.bytes_h2d += counters[1] * g.expert_bytes;
                ggml_backend_buffer_free(g.lru_buf);
                sycl::free(g.lru_host, moe_cache_queue(device));
                mc.groups.erase(mc.groups.begin() + gi);
            }
            if (mc.groups.empty()) {
                if (mc.ops > 0) {
                    GGML_LOG_INFO("%s: SYCL%d MoE cache: %llu ops (%llu staged), %llu expert hits, %llu misses (%.1f%% hit), "
                                  "%.2f GiB host to device, %.2f GiB device to device\n",
                                  __func__, device, (unsigned long long) mc.ops, (unsigned long long) mc.staged_ops,
                                  (unsigned long long) mc.hits, (unsigned long long) mc.misses,
                                  100.0 * mc.hits / std::max<uint64_t>(1, mc.hits + mc.misses),
                                  mc.bytes_h2d / double(1ull << 30), mc.bytes_d2d / double(1ull << 30));
                }
                if (mc.scratch) {
                    ggml_backend_buffer_free(mc.scratch);
                }
                mc = moe_cache_device{};
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

// Packs the experts of a batch routed to more experts than there are slots into the scratch, in expert order,
// copying resident experts from their slots and the rest from host memory in contiguous runs.
static void moe_cache_stage(queue_ptr stream, moe_cache_device & mc, moe_cache_group & g, std::vector<int32_t> & need,
                            std::vector<int32_t> & packed) {
    std::sort(need.begin(), need.end());
    g.n_packed = (int64_t) need.size();
    packed.assign(g.n_expert, -1);
    for (size_t p = 0; p < need.size(); ++p) {
        packed[need[p]] = (int32_t) p;
    }
    char * scratch = (char *) ggml_backend_buffer_get_base(mc.scratch);
    for (const moe_cache_member & m : g.members) {
        const size_t nb2  = m.tensor->nb[2];
        char *       base = scratch + m.scratch_off;
        const char * pool = (const char *) ggml_backend_buffer_get_base(m.pool);
        const char * host = (const char *) m.tensor->data;
        for (size_t p = 0; p < need.size();) {
            const int32_t e = need[p];
            if (g.slot_of_expert[e] >= 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(base + p * nb2, pool + g.slot_of_expert[e] * nb2, nb2)));
                mc.bytes_d2d += nb2;
                p++;
                continue;
            }
            size_t run = 1;
            while (p + run < need.size() && need[p + run] == e + (int32_t) run && g.slot_of_expert[need[p + run]] < 0) {
                run++;
            }
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(base + p * nb2, host + e * nb2, run * nb2)));
            mc.bytes_h2d += run * nb2;
            p += run;
        }
    }
    mc.staged_ops++;
}

// Gives every expert of the batch a slot, evicting the least recently used experts not needed by this batch,
// and copies the missing ones from host memory.
static void moe_cache_assign(queue_ptr stream, moe_cache_device & mc, moe_cache_group & g,
                             const std::vector<int32_t> & need) {
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
}

static void moe_cache_lru_pack(moe_cache_group & g) {
    const moe_cache_lru_layout l = moe_cache_lru_layout_of(g);
    memcpy(g.lru_host, g.slot_of_expert.data(), g.n_expert * sizeof(int32_t));
    memcpy(g.lru_host + l.expert_of_slot, g.expert_of_slot.data(), g.n_slots * sizeof(int32_t));
    memcpy(g.lru_host + l.last_used, g.last_used.data(), g.n_slots * sizeof(uint32_t));
    memcpy(g.lru_host + l.tick, &g.tick, sizeof(uint32_t));
}

static void moe_cache_lru_unpack(moe_cache_group & g) {
    const moe_cache_lru_layout l = moe_cache_lru_layout_of(g);
    memcpy(g.slot_of_expert.data(), g.lru_host, g.n_expert * sizeof(int32_t));
    memcpy(g.expert_of_slot.data(), g.lru_host + l.expert_of_slot, g.n_slots * sizeof(int32_t));
    memcpy(g.last_used.data(), g.lru_host + l.last_used, g.n_slots * sizeof(uint32_t));
    memcpy(&g.tick, g.lru_host + l.tick, sizeof(uint32_t));
}

// A batch is planned on the device when it can never route to more experts than there are slots, so it is never
// staged, and every member's experts can be copied in 16-byte chunks.
static bool moe_cache_plans_on_device(const moe_cache_group & g, size_t n) {
    if (!ggml_sycl_moe_cache_device_plan() || g.members.size() > MOE_CACHE_MEMBERS) {
        return false;
    }
    if ((int64_t) n > g.n_slots && g.n_slots < g.n_expert) {
        return false;
    }
    for (const moe_cache_member & m : g.members) {
        if (m.tensor->nb[2] % sizeof(sycl::uint4) != 0 || (uintptr_t) m.tensor->data % sizeof(sycl::uint4) != 0) {
            return false;
        }
    }
    return true;
}

struct moe_cache_copy_args {
    const sycl::uint4 * host[MOE_CACHE_MEMBERS];
    sycl::uint4 *       pool[MOE_CACHE_MEMBERS];
    size_t              chunks[MOE_CACHE_MEMBERS];
    int                 n;
};

// Runs the moe_cache_assign LRU on the device against the group's device LRU copy. One work-group loads the LRU
// into local memory, marks hits and collects the missing experts in parallel, gives the n-th missing expert the
// eligible slot of rank n by (last_used, slot), which is the slot the sequential host LRU would evict n-th, then
// writes the remapped ids, the miss list and the LRU back. A grid-stride kernel then copies the missing experts of
// every member straight out of pinned host memory. Kernel reads of pinned memory run at copy engine speed, and the
// host never waits for the routing.
static void moe_cache_plan_device(queue_ptr stream, moe_cache_group & g, const ggml_tensor * ids) {
    using local_atomic = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;

    const moe_cache_lru_layout l   = moe_cache_lru_layout_of(g);
    int32_t *                  lru = (int32_t *) ggml_backend_buffer_get_base(g.lru_buf);
    if (g.host_dirty) {
        moe_cache_lru_pack(g);
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(lru, g.lru_host, l.state_words * sizeof(int32_t))));
        g.host_dirty = false;
    }

    const char *  ids_d    = (const char *) ids->data;
    const size_t  nb0      = ids->nb[0];
    const size_t  nb1      = ids->nb[1];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_ids    = n_used * ids->ne[1];
    const int64_t n_expert = g.n_expert;
    const int     n_slots  = g.n_slots;
    int32_t *     remap    = (int32_t *) ggml_backend_buffer_get_base(g.ids_buf);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1> smem(sycl::range<1>(2 * n_expert + 4 * n_slots + 2), cgh);
        cgh.parallel_for(sycl::nd_range<1>(MOE_CACHE_PLAN_WG, MOE_CACHE_PLAN_WG), [=](sycl::nd_item<1> it) {
            const int      tid            = it.get_local_id(0);
            const int      nth            = it.get_local_range(0);
            int32_t *      slot_of_expert = smem.get_multi_ptr<sycl::access::decorated::no>().get();
            int32_t *      seen           = slot_of_expert + n_expert;
            int32_t *      expert_of_slot = seen + n_expert;
            uint32_t *     last_used      = (uint32_t *) (expert_of_slot + n_slots);
            int32_t *      missing        = (int32_t *) (last_used + n_slots);
            int32_t *      victim         = missing + n_slots;
            int32_t *      counts         = victim + n_slots;
            uint32_t *     lru_last_used  = (uint32_t *) (lru + l.last_used);
            uint64_t *     counters       = (uint64_t *) (lru + l.counters);
            const uint32_t tick           = *(const uint32_t *) (lru + l.tick) + 1;
            auto           id_at          = [&](int64_t i) {
                return *(const int32_t *) (ids_d + (i / n_used) * nb1 + (i % n_used) * nb0);
            };
            auto key_of = [&](int s) {
                return ((uint64_t) last_used[s] << 32) | (uint32_t) s;
            };

            for (int64_t e = tid; e < n_expert; e += nth) {
                slot_of_expert[e] = lru[e];
                seen[e]           = 0;
            }
            for (int s = tid; s < n_slots; s += nth) {
                expert_of_slot[s] = lru[l.expert_of_slot + s];
                last_used[s]      = lru_last_used[s];
            }
            if (tid < 2) {
                counts[tid] = 0;
            }
            sycl::group_barrier(it.get_group());

            for (int64_t i = tid; i < n_ids; i += nth) {
                const int32_t e = id_at(i);
                if (local_atomic(seen[e]).exchange(1) != 0) {
                    continue;
                }
                const int32_t s = slot_of_expert[e];
                if (s >= 0) {
                    last_used[s] = tick;
                    local_atomic(counts[0]).fetch_add(1);
                } else {
                    missing[local_atomic(counts[1]).fetch_add(1)] = e;
                }
            }
            sycl::group_barrier(it.get_group());

            const int n_miss = counts[1];
            if (n_miss > 0) {
                for (int s = tid; s < n_slots; s += nth) {
                    if (last_used[s] == tick) {
                        continue;
                    }
                    const uint64_t key  = key_of(s);
                    int            rank = 0;
                    for (int t = 0; t < n_slots && rank < n_miss; ++t) {
                        rank += last_used[t] != tick && key_of(t) < key;
                    }
                    if (rank < n_miss) {
                        victim[rank] = s;
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int k = tid; k < n_miss; k += nth) {
                    const int32_t s   = victim[k];
                    const int32_t e   = missing[k];
                    const int32_t old = expert_of_slot[s];
                    if (old >= 0) {
                        slot_of_expert[old] = -1;
                    }
                    slot_of_expert[e]      = s;
                    expert_of_slot[s]      = e;
                    last_used[s]           = tick;
                    lru[l.miss_slot + k]   = s;
                    lru[l.miss_expert + k] = e;
                }
                sycl::group_barrier(it.get_group());
            }

            for (int64_t i = tid; i < n_ids; i += nth) {
                remap[i] = slot_of_expert[id_at(i)];
            }
            for (int64_t e = tid; e < n_expert; e += nth) {
                lru[e] = slot_of_expert[e];
            }
            for (int s = tid; s < n_slots; s += nth) {
                lru[l.expert_of_slot + s] = expert_of_slot[s];
                lru_last_used[s]          = last_used[s];
            }
            if (tid == 0) {
                *(uint32_t *) (lru + l.tick) = tick;
                lru[l.miss_count]            = n_miss;
                counters[0] += counts[0];
                counters[1] += n_miss;
            }
        });
    });

    moe_cache_copy_args a{};
    for (const moe_cache_member & m : g.members) {
        a.host[a.n]   = (const sycl::uint4 *) m.tensor->data;
        a.pool[a.n]   = (sycl::uint4 *) ggml_backend_buffer_get_base(m.pool);
        a.chunks[a.n] = m.tensor->nb[2] / sizeof(sycl::uint4);
        a.n++;
    }
    stream->parallel_for(
        sycl::nd_range<1>(MOE_CACHE_COPY_WGS * MOE_CACHE_COPY_WG, MOE_CACHE_COPY_WG), [=](sycl::nd_item<1> it) {
            const size_t n_miss = lru[l.miss_count];
            for (int k = 0; k < a.n; ++k) {
                const size_t chunks = a.chunks[k];
                for (size_t i = it.get_global_id(0); i < n_miss * chunks; i += it.get_global_range(0)) {
                    const size_t m = i / chunks;
                    const size_t c = i - m * chunks;
                    a.pool[k][(size_t) lru[l.miss_slot + m] * chunks + c] =
                        a.host[k][(size_t) lru[l.miss_expert + m] * chunks + c];
                }
            }
        });
    g.device_lru = true;
}

// Plans the layer's routing on the device when the batch can never be staged (see moe_cache_plan_device).
// Otherwise reads the routing on the host and either gives every routed expert a slot or, when the batch needs
// more experts than there are slots, stages them. The queue is drained before the ids are copied back: reading
// them right behind the routing kernels returned stale ids in about one of three Coder IQ4_XS decode runs, a
// race standalone tests of the same queue pattern did not reproduce.
static void moe_cache_plan(ggml_backend_sycl_context & ctx, moe_cache_device & mc, moe_cache_group & g,
                           const ggml_tensor * ids, ggml_sycl_moe_cache_reorder_t reorder) {
    const queue_ptr stream = ctx.stream();
    for (moe_cache_member & m : g.members) {
        reorder(&ctx, m.tensor);
        m.used = false;
    }

    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const size_t  n        = n_used * n_tokens;
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
    g.plan_ids = ids->data;
    if (moe_cache_plans_on_device(g, n)) {
        g.staged = false;
        moe_cache_plan_device(stream, g, ids);
        return;
    }

    std::vector<char> ids_raw(ggml_nbytes(ids));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
    if (g.device_lru) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(g.lru_host, ggml_backend_buffer_get_base(g.lru_buf),
                                                  moe_cache_lru_layout_of(g).state_words * sizeof(int32_t))));
    }
    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(ids_raw.data(), ids->data, ids_raw.size()).wait()));
    if (g.device_lru) {
        moe_cache_lru_unpack(g);
        g.device_lru = false;
    }
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
    std::vector<int32_t> packed;
    g.tick++;
    g.staged = (int) need.size() > g.n_slots;
    if (g.staged) {
        moe_cache_stage(stream, mc, g, need, packed);
    } else {
        moe_cache_assign(stream, mc, g, need);
    }
    g.host_dirty = true;

    const std::vector<int32_t> & remap = g.staged ? packed : g.slot_of_expert;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            g.ids_host[t * n_used + u] = remap[id_at(t, u)];
        }
    }
    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ggml_backend_buffer_get_base(g.ids_buf), g.ids_host, n * sizeof(int32_t))));
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

    src0_view = *src0;
    if (g->staged) {
        src0_view.data   = (char *) ggml_backend_buffer_get_base(mc.scratch) + m.scratch_off;
        src0_view.buffer = mc.scratch;
        src0_view.ne[2]  = g->n_packed;
    } else {
        src0_view.data   = ggml_backend_buffer_get_base(m.pool);
        src0_view.buffer = m.pool;
        src0_view.ne[2]  = g->n_slots;
    }
    src0_view.nb[3]     = src0->nb[2] * src0_view.ne[2];
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
