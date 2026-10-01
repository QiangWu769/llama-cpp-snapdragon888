#pragma once

#include <stdbool.h>
#include <stddef.h>

/* Optional whole-tensor model/context buffer packing cap, in MiB. Keep the
 * established 256 MiB default. Generic graph allocations can bypass this
 * hook; the packed-output cache also has a separate size contract. */
static inline bool ggml_htp_parse_max_buffer_mb(const char *value, size_t *bytes) {
    if (!bytes) return false;
    if (!value) {
        *bytes = 256UL * 1024 * 1024;
        return true;
    }
    if (!value[0]) return false;
    size_t mib = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || mib > (256 - size_t(*p - '0')) / 10) return false;
        mib = mib * 10 + size_t(*p - '0');
    }
    if (!mib) return false;
    *bytes = mib * 1024UL * 1024;
    return true;
}
