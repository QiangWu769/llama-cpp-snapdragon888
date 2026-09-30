#include "htp-ops.h"
#include "ggml-htp-op-support.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "ggml-backend-impl.h"
#include "dsprpc_interface.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml-htp-output-pack.h"
#include "ggml.h"

////// Special headers intended for CPU-NPU communication. Keep them in sync with ops backend.
#include "message.h"
#include "op_reg.h"

namespace {

auto get_all_rpcmem_mappings(const ggml_tensor * dst) {
    const auto & mapper = ggml_backend_htp_context::instance()->mapper;

    std::vector<std::pair<int, ssize_t>> mappings;
    if (ggml_backend_buft_is_rpcmem(dst->buffer->buft)) {
        mappings.push_back(mapper.get_tensor_mapping(dst));
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src && ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
            mappings.push_back(mapper.get_tensor_mapping(src));
        }
    }
    return mappings;
}

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void write_buf(uint8_t *& p, void * src, size_t size) {
    std::memcpy((void *) p, src, size);
    p += size;
}

uint8_t param_buf[4096];  // TODO(hzx): better implementation
std::mutex request_mutex;

bool env_enabled(const char * name) {
    const char * value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

int64_t op_timeout_ms() {
    static const int64_t timeout = []() -> int64_t {
        const char * value = getenv("HTP_OP_TIMEOUT_MS");
        if (!value) {
            return 120000;
        }
        char * end = nullptr;
        errno = 0;
        const long long parsed = strtoll(value, &end, 10);
        if (errno || end == value || *end || parsed <= 0 || parsed > 3600000) {
            GGML_ABORT("HTP_OP_TIMEOUT_MS must be an integer between 1 and 3600000");
        }
        return parsed;
    }();
    return timeout;
}

bool rpcmem_tensor(const ggml_tensor * tensor) {
    return tensor && tensor->buffer && ggml_backend_buft_is_rpcmem(tensor->buffer->buft);
}

}  // namespace

extern "C" {

bool htp_ops_has_permuted_weight(const struct ggml_tensor *dst) {
    return ggml_htp_has_permuted_weight(dst);
}

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    auto * ctx = ggml_backend_htp_context::instance();
    if (ctx->skip_htp_ops) {
        return false;
    }
    if (!ctx->ops_backend_initialized) {
        return false;
    }

    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            return false;

            if (dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32) {
                // NOTE: RPC version is mainly for testing
                return dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32") != nullptr;
            }
            return false;
        case GGML_OP_MUL_MAT:
            return ggml_htp_op_layout_supported(dst) && rpcmem_tensor(dst->src[0]) &&
                   rpcmem_tensor(dst->src[1]) && rpcmem_tensor(dst);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_htp_op_layout_supported(dst) && rpcmem_tensor(dst->src[0]) &&
                   rpcmem_tensor(dst->src[1]) && rpcmem_tensor(dst->src[2]) &&
                   rpcmem_tensor(dst->src[3]) && rpcmem_tensor(dst);
        default:
            return false;
    }
}

int htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return 0;
    }

    // The context, message channel, parameter buffer, and mapper are shared.
    std::lock_guard<std::mutex> lock(request_mutex);
    GGML_ASSERT(htp_ops_support_op(dst));
    const auto started = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(op_timeout_ms());

    prepare_tensor_rpcmem_mapping(dst);

    auto * ctx           = ggml_backend_htp_context::instance();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    constexpr bool prefer_rpc = false;

    int op_index  = -1;
    int args_size = 0;  // strictly 32 bits

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 2);

                auto [dst_fd, dst_offset] = mappings[0];
                auto [src_fd, src_offset] = mappings[1];

                if (prefer_rpc) {
                    using fn_type = int(int, int, int, int, int, int);

                    auto op_fn = reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32"));
                    GGML_ASSERT(op_fn);

                    return op_fn(dst_fd, dst_offset, src_fd, src_offset, dst->ne[0], ggml_nrows(dst));
                }

                RmsNormF32Params params{
                    .dst = { dst_fd, (int32_t) dst_offset },
                    .src = { src_fd, (int32_t) src_offset },
                    .ne0 = (int32_t) dst->ne[0],
                    .ne1 = (int32_t) ggml_nrows(dst),
                };
                *reinterpret_cast<RmsNormF32Params *>(param_buf) = params;

                op_index  = HTP_OPS_RMS_NORM_F32;
                args_size = sizeof(RmsNormF32Params);
            }
            break;

        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 3);

                auto [output_fd, output_offset]         = mappings[0];
                auto [weight_fd, weight_offset]         = mappings[1];
                auto [activation_fd, activation_offset] = mappings[2];

                int m = ggml_nrows(activation);
                int k = weight->ne[0];
                int n = weight->ne[1];

                MatMulParams params{
                    .output     = { output_fd,     (int32_t) output_offset     },
                    .activation = { activation_fd, (int32_t) activation_offset },
                    .weight     = { weight_fd,     (int32_t) weight_offset     },
                    .m          = m,
                    .k          = k,
                    .n          = n,
                };
                *reinterpret_cast<MatMulParams *>(param_buf) = params;

                args_size = sizeof(MatMulParams);

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    if (prefer_rpc) {
                        using fn_type = int(int, int, int, int, int, int, int, int, int);

                        auto op_fn =
                            reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32"));
                        GGML_ASSERT(op_fn);

                        return op_fn(output_fd, output_offset, activation_fd, activation_offset, weight_fd,
                                     weight_offset, m, k, n);
                    }

                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
                } else {
                    GGML_ASSERT(false && "not implemented");
                }
            }
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            {
                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 5);

                auto [o_fd, o_offset]       = mappings[0];
                auto [q_fd, q_offset]       = mappings[1];
                auto [k_fd, k_offset]       = mappings[2];
                auto [v_fd, v_offset]       = mappings[3];
                auto [mask_fd, mask_offset] = mappings[4];

                int head_dim   = q->ne[0];
                int qo_len     = q->ne[1];
                int kv_len     = k->ne[1];
                int n_heads    = q->ne[2];
                int n_kv_heads = k->ne[2];

                FlashAttnParams params{
                    .o          = { o_fd,    (int32_t) o_offset    },
                    .q          = { q_fd,    (int32_t) q_offset    },
                    .k          = { k_fd,    (int32_t) k_offset    },
                    .v          = { v_fd,    (int32_t) v_offset    },
                    .mask       = { mask_fd, (int32_t) mask_offset },
                    .qo_len     = qo_len,
                    .kv_len     = kv_len,
                    .n_heads    = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim   = head_dim,
                };
                *reinterpret_cast<FlashAttnParams *>(param_buf) = params;

                op_index  = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                args_size = sizeof(FlashAttnParams);
            }
            break;

        default:
            break;
    }

    // TODO(hzx): make sure only one thread can arrive here
    int  n_reqs         = 1;
    int  n_unmap_fds    = ctx->mapper.get_pending_unmap_reqs().size();
    bool has_unmap_reqs = n_unmap_fds > 0;
    if (has_unmap_reqs) {
        ++n_reqs;
    }

    size_t op_req_size = sizeof(RequestHeader) + sizeof(OpComputeRequest) + args_size;

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);

    // FIXME: this is very ugly
    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    // std::atomic_store_explicit(d_ptr, 0, std::memory_order_release);

    // The memory order here is not very important
    std::atomic_store(d_ptr, 0);

    msg_hdr->n_reqs         = n_reqs;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + op_req_size;

    {
        RequestHeader req_hdr{
            .state = INT32_MIN,
            .type  = REQUEST_TYPE_OP_COMPUTE,
        };
        OpComputeRequest op_req{
            .op = (uint32_t) op_index,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, op_req);
        write_buf(p, param_buf, args_size);
    }

    if (has_unmap_reqs) {
        size_t map_req_size     = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t);
        msg_hdr->req_offsets[2] = msg_hdr->req_offsets[1] + map_req_size;

        RequestHeader req_hdr{
            .state = INT32_MIN,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_unmap_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 1));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (const auto & [fd, _base, _len] : ctx->mapper.get_pending_unmap_reqs()) {
            write_buf(p, fd);
        }
    }

    // compute checksum
    if (1) {
        uint32_t   sum   = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;  // skip state & checksum
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;

        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;  // value of `state`

        msg_hdr->checksum = -sum;
    } else {
#ifdef __aarch64__
        asm volatile("dmb sy" ::: "memory");
#endif
    }

    // issue request
    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));

    // NOTE(hzx): make sure memory_order_release is used here to ensure all previous writes are valid
    std::atomic_store_explicit(v0_ptr, 1, std::memory_order_release);

    // poll for response
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        if (std::chrono::steady_clock::now() - started >= timeout) {
            GGML_ABORT("HTP: DSP request timed out after %lld ms, tensor=%s op=%s opcode=%d; "
                       "aborting without reusing buffers still owned by DSP",
                       static_cast<long long>(op_timeout_ms()), dst->name, ggml_op_name(dst->op), op_index);
        }
        // TODO(hzx): use cpu_relax here
        usleep(1);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    for (int i = 0; i < n_reqs; ++i) {
        const int status = message_header_get_request_ptr(msg_hdr, i)->state;
        if (status != 0) {
            GGML_ABORT("HTP: DSP request failed, tensor=%s op=%s opcode=%d request=%d status=%d",
                       dst->name, ggml_op_name(dst->op), op_index, i, status);
        }
    }
    d_ptr->store(0, std::memory_order_relaxed);

    if (has_unmap_reqs) {
        ctx->mapper.unmap_all_pending_buffers();
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    if (env_enabled("HTP_TRACE")) {
        static uint64_t counts[HTP_OPS_COUNT] = {};
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count();
        ++counts[op_index];
        fprintf(stderr, "HTP DSP completed: opcode=%d count=%llu tensor=%s op=%s elapsed_us=%lld status=0\n",
                op_index, static_cast<unsigned long long>(counts[op_index]), dst->name, ggml_op_name(dst->op),
                static_cast<long long>(elapsed_us));
        if (dst->op == GGML_OP_MUL_MAT) {
            fprintf(stderr, "HTP matrix shape: tensor=%s M=%lld K=%lld N=%lld\n", dst->name,
                    static_cast<long long>(ggml_nrows(dst->src[1])),
                    static_cast<long long>(dst->src[0]->ne[0]),
                    static_cast<long long>(dst->src[0]->ne[1]));
        }
    }
    return 0;
}

void htp_ops_retire_and_free_rpcmem(void *base) {
    std::lock_guard<std::mutex> lock(request_mutex);
    auto *ctx = ggml_backend_htp_context::get_if_initialized();
    RpcMemMapper::UnmapRequest mapping;
    if (ctx && ctx->mapper.get_buffer_mapping(base, mapping)) {
        auto [fd, address, size] = mapping;
        GGML_UNUSED(address);
        GGML_UNUSED(size);
        auto *msg = static_cast<MessageHeader *>(ctx->ops_msg_chan);
        // No operation is in flight while request_mutex is held. Replace its
        // payload with a MAP-only job while the published state is zero.
        auto *state = reinterpret_cast<volatile std::atomic<uint64_t> *>(msg);
        state->store(0, std::memory_order_release);
        std::memset(reinterpret_cast<uint8_t *>(msg) + sizeof(msg->state), 0,
                    ggml_backend_htp_context::MAX_MSG_SIZE - sizeof(msg->state));
        msg->n_reqs = 1;
        msg->req_offsets[0] = message_header_size(msg);
        msg->req_offsets[1] = msg->req_offsets[0] + sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + sizeof(int32_t);
        RequestHeader request { .state = INT32_MIN, .type = REQUEST_TYPE_RPCMEM_MAP };
        RpcmemMapRequest payload { .n_puts = 1, .n_gets = 0 };
        auto *cursor = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg, 0));
        write_buf(cursor, request);
        write_buf(cursor, payload);
        write_buf(cursor, static_cast<int32_t>(fd));
        uint32_t sum = 1;
        auto *words = reinterpret_cast<uint32_t *>(msg);
        for (size_t i = 3; i < ggml_backend_htp_context::MAX_MSG_SIZE / 4; ++i) sum += words[i];
        msg->checksum = -sum;
        const auto started = std::chrono::steady_clock::now();
        auto *ready = reinterpret_cast<volatile std::atomic<uint8_t> *>(&msg->state.v[0]);
        auto *done = reinterpret_cast<volatile std::atomic<uint8_t> *>(&msg->state.v[1]);
        ready->store(1, std::memory_order_release);
        while (done->load(std::memory_order_acquire) == 0) {
            if (std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(op_timeout_ms())) {
                GGML_ABORT("HTP buffer retirement timed out; mapped buffer retained");
            }
            usleep(1);
        }
        const int status = message_header_get_request_ptr(msg, 0)->state;
        if (status != 0) {
            GGML_ABORT("HTP buffer retirement: DSP HAP_mmap_put failed status=%d; buffer retained", status);
        }
        state->store(0, std::memory_order_release);
        ctx->mapper.retire_buffer_mapping(base);
    }
    rpcmem_free(base);
}
}
