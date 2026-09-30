#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <tuple>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "rpcmem_mapper.h"

static ggml_backend_buffer_type rpcmem_type {};
static ggml_backend_buffer_type cpu_type {};
static int map_calls;
static int unmap_calls;
static int fail_unmap_fd = -1;
static int last_unmapped_fd = -1;

extern "C" bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t type) { return type == &rpcmem_type; }
extern "C" void *ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) { return buffer->context; }
extern "C" const char *ggml_op_name(ggml_op) { return "test"; }
extern "C" int rpcmem_to_fd(void *address) { return int(uintptr_t(address) / 4096); }
extern "C" int fastrpc_mmap(int domain, int, void *, int offset, size_t, fastrpc_map_flags flags) {
    assert(domain == CDSP_DOMAIN_ID && offset == 0 && flags == FASTRPC_MAP_FD);
    ++map_calls;
    return 0;
}
extern "C" int fastrpc_munmap(int domain, int fd, void *address, size_t length) {
    assert(domain == CDSP_DOMAIN_ID && fd == rpcmem_to_fd(address) && length > 0);
    ++unmap_calls;
    last_unmapped_fd = fd;
    return fd == fail_unmap_fd ? 19 : 0;
}
extern "C" void ggml_abort(const char *, int, const char *, ...) { throw std::runtime_error("expected abort"); }
ggml_backend_htp_context *ggml_backend_htp_context::instance() { throw std::runtime_error("unexpected singleton initialization"); }

int main() {
    ggml_backend_buffer a {}, b {}, cpu {};
    a.buft = b.buft = &rpcmem_type;
    cpu.buft = &cpu_type;
    a.context = reinterpret_cast<void *>(uintptr_t(11 * 4096));
    b.context = reinterpret_cast<void *>(uintptr_t(12 * 4096));
    a.size = 512;
    b.size = 1024;
    ggml_tensor target {};
    target.buffer = &a;
    RpcMemMapper mapper(1024, true);
    RpcMemMapper::UnmapRequest found;
    mapper.validate(&target);
    assert(map_calls == 1 && mapper.get_buffer_mapping(a.context, found));
    assert(std::get<0>(found) == 11 && std::get<2>(found) == 512);
    target.buffer = &b;
    mapper.validate(&target);  // evict A to pending while mapping B
    assert(map_calls == 2 && mapper.get_pending_unmap_reqs().size() == 1);
    assert(mapper.get_buffer_mapping(a.context, found));
    fail_unmap_fd = 11;
    bool pending_failed = false;
    try { mapper.unmap_all_pending_buffers(); }
    catch (const std::runtime_error &) { pending_failed = true; }
    assert(pending_failed && unmap_calls == 1 && mapper.get_pending_unmap_reqs().size() == 1);
    assert(mapper.get_buffer_mapping(a.context, found));
    fail_unmap_fd = -1;
    mapper.retire_buffer_mapping(a.context);
    assert(unmap_calls == 2 && last_unmapped_fd == 11);
    assert(mapper.get_pending_unmap_reqs().empty() && !mapper.get_buffer_mapping(a.context, found));
    assert(mapper.get_buffer_mapping(b.context, found));
    fail_unmap_fd = 12;
    bool failed = false;
    try { mapper.retire_buffer_mapping(b.context); }
    catch (const std::runtime_error &) { failed = true; }
    assert(failed && unmap_calls == 3 && mapper.get_buffer_mapping(b.context, found));
    fail_unmap_fd = -1;
    mapper.retire_buffer_mapping(b.context);
    assert(unmap_calls == 4 && !mapper.get_buffer_mapping(b.context, found));
    mapper.retire_buffer_mapping(b.context);  // already retired is a no-op
    assert(unmap_calls == 4);
    target.buffer = &a;
    mapper.validate(&target);  // retirement removed active/LRU state correctly
    assert(map_calls == 3 && mapper.get_buffer_mapping(a.context, found));
    mapper.retire_buffer_mapping(a.context);
    assert(unmap_calls == 5);
    puts("Host active/pending mapping retirement test PASS");
}
