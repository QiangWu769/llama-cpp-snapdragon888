// This is a C++ header
#pragma once

#include "ggml.h"
#include "htp-cpu-impl.h"
#include "rpcmem_mapper.h"

#include <mutex>

static const char * HTP_OPS_DL_PATH = "libhtp_ops.so";

// singleton HTP backend context
struct ggml_backend_htp_context {
    static constexpr size_t MAX_MSG_SIZE = 4096;

    // stuff in struct ggml_backend_cpu_context, see ggml-cpu.cpp
    size_t    work_size = 0;
    uint8_t * work_data = nullptr;

    int                      n_threads  = 0;
    struct ggml_threadpool * threadpool = nullptr;
    bool reuse_threadpool = false;

    // The singleton shares its work buffer and pool across backend wrappers.
    // Serialize graph execution with changes to the pool's thread count.
    std::mutex compute_mutex;

    // TODO(hzx): add abort_callback & abort_callback_data

    // shared rpcmem mapper
    RpcMemMapper mapper;

    // HTP ops backend library
    void * ops_dl_handle;
    bool   ops_backend_initialized = false;
    void * ops_msg_chan            = nullptr;
    int    msg_chan_fd;

    // debug
    bool skip_htp_ops = false;

    ggml_backend_htp_context();
    ~ggml_backend_htp_context();

    int init_message_channel();

    static ggml_backend_htp_context * instance();
    static ggml_backend_htp_context * get_if_initialized();
};
