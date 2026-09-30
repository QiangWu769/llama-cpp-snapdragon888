#include <cassert>
#include <cstdint>
#include <cstdio>

#include "dsp/mmap_mgr.h"

static int get_calls;
static int put_calls;
static int fail_get_fd = -1;
static int fail_put_fd = -1;

extern "C" int HAP_mmap_get(int fd, void **address, size_t *) {
    ++get_calls;
    if (fd == fail_get_fd) return 9;
    *address = reinterpret_cast<void *>(uintptr_t(0x100000 + fd * 4096));
    return 0;
}
extern "C" int HAP_mmap_put(int fd) {
    ++put_calls;
    return fd == fail_put_fd ? 17 : 0;
}

int main() {
    void *address = mmap_manager_get_map(11);
    assert(address != nullptr && get_calls == 1);
    assert(mmap_manager_get_map(11) == address && get_calls == 1);
    fail_put_fd = 11;
    assert(mmap_manager_put_map(11) == 17 && put_calls == 1);
    // A failed HAP_mmap_put must retain the acquired mapping and reference.
    assert(mmap_manager_get_map(11) == address && get_calls == 1);
    fail_put_fd = -1;
    assert(mmap_manager_put_map(11) == 0 && put_calls == 2);
    assert(mmap_manager_get_map(11) == address && get_calls == 2);
    assert(mmap_manager_put_map(11) == 0 && put_calls == 3);
    assert(mmap_manager_put_map(11) == 0 && put_calls == 3);
    assert(mmap_manager_put_map(99) == 0 && put_calls == 3);
    fail_get_fd = 12;
    assert(mmap_manager_get_map(12) == nullptr && get_calls == 3);
    assert(mmap_manager_put_map(12) == 0 && put_calls == 3);
    fail_get_fd = -1;
    assert(mmap_manager_get_map(12) != nullptr && get_calls == 4);
    assert(mmap_manager_put_map(12) == 0 && put_calls == 4);
    puts("DSP mapping retention/retry test PASS");
}
