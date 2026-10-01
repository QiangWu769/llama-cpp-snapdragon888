#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "rpcmem_mapper.h"
#include "htp-buffer-config.h"

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

static void budget_configuration_test() {
    const char *saved = std::getenv("GGML_HTP_MAP_BUDGET_MB");
    const bool had_saved = saved != nullptr;
    const std::string saved_value = saved ? saved : "";
    assert(unsetenv("GGML_HTP_MAP_BUDGET_MB") == 0);
    const size_t mib = 1024UL * 1024;
    assert(RpcMemMapper::configured_map_budget() == 3 * 1024UL * mib);
    for (const char *value : {"512", "768", "3072", "000512"}) {
        assert(setenv("GGML_HTP_MAP_BUDGET_MB", value, 1) == 0);
        assert(RpcMemMapper::configured_map_budget() == size_t(std::stoul(value)) * mib);
    }
    const std::string overflow = std::to_string(std::numeric_limits<size_t>::max() / mib + 1);
    for (const std::string &value : {std::string(""), std::string("0"), std::string("-1"),
         std::string("+512"), std::string(" 512"), std::string("1.5"), std::string("512x"), overflow}) {
        assert(setenv("GGML_HTP_MAP_BUDGET_MB", value.c_str(), 1) == 0);
        bool failed = false;
        try { (void) RpcMemMapper::configured_map_budget(); }
        catch (const std::runtime_error &) { failed = true; }
        assert(failed);
    }
    if (had_saved) assert(setenv("GGML_HTP_MAP_BUDGET_MB", saved_value.c_str(), 1) == 0);
    else assert(unsetenv("GGML_HTP_MAP_BUDGET_MB") == 0);
}

static void buffer_configuration_test() {
    const size_t mib = 1024UL * 1024;
    size_t bytes = 0;
    assert(ggml_htp_parse_max_buffer_mb(nullptr, &bytes) && bytes == 256 * mib);
    for (const char *value : {"1", "64", "128", "255", "256", "00064"}) {
        assert(ggml_htp_parse_max_buffer_mb(value, &bytes));
        assert(bytes == size_t(std::stoul(value)) * mib);
    }
    for (const char *value : {"", "0", "-1", "+64", " 64", "64 ", "64.0", "257",
                              "999999999999999999999999999999999999"}) {
        bytes = 123;
        assert(!ggml_htp_parse_max_buffer_mb(value, &bytes) && bytes == 123);
    }
    assert(!ggml_htp_parse_max_buffer_mb("64", nullptr));
}

int main() {
    budget_configuration_test();
    buffer_configuration_test();
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

    // Duplicate sources/views share one mapping and one budget contribution.
    ggml_tensor source {};
    source.buffer = &a;
    target.buffer = &a;
    target.src[0] = target.src[1] = &source;
    mapper.validate(&target);
    assert(map_calls == 4 && mapper.get_pending_unmap_reqs().empty());

    // A cached source plus an uncached output must fit as a complete set.
    // Reject before eviction/map mutation rather than dropping the source.
    target.buffer = &b;
    const int maps_before = map_calls;
    bool insufficient = false;
    try { mapper.validate(&target); }
    catch (const std::runtime_error &) { insufficient = true; }
    assert(insufficient && map_calls == maps_before && mapper.get_pending_unmap_reqs().empty());
    assert(mapper.get_buffer_mapping(a.context, found) && !mapper.get_buffer_mapping(b.context, found));
    mapper.retire_buffer_mapping(a.context);

    // An exact-budget unique set succeeds; the next operation evicts only a
    // no-longer-needed LRU buffer and keeps the pending protocol intact.
    RpcMemMapper exact(1536, true);
    target.buffer = &b;
    exact.validate(&target);
    assert(exact.get_buffer_mapping(a.context, found) && exact.get_buffer_mapping(b.context, found));
    ggml_backend_buffer c {};
    c.buft = &rpcmem_type;
    c.context = reinterpret_cast<void *>(uintptr_t(13 * 4096));
    c.size = 512;
    target.buffer = &c;
    target.src[0] = target.src[1] = nullptr;
    exact.validate(&target);
    assert(exact.get_pending_unmap_reqs().size() == 1);
    assert(std::get<1>(exact.get_pending_unmap_reqs().front()) == a.context);
    assert(exact.get_buffer_mapping(b.context, found) && exact.get_buffer_mapping(c.context, found));
    exact.unmap_all_pending_buffers();
    assert(exact.get_pending_unmap_reqs().empty());
    exact.retire_buffer_mapping(b.context);
    exact.retire_buffer_mapping(c.context);
    puts("Host mapper PASS: map/allocation budget parsing, unique working sets, insufficient-budget rejection, LRU and active/pending retirement");
}
