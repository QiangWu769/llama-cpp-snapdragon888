#pragma once

// Test-owned substitutes for the two SDK functions used by mmap_mgr.cc.
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int HAP_mmap_get(int fd, void **address, size_t *length);
int HAP_mmap_put(int fd);
#ifdef __cplusplus
}
#endif
