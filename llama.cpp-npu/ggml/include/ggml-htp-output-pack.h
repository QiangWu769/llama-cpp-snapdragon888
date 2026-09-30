#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Experimental runtime-owned output caches. Ordinary token embeddings retain
 * their original row-major layout and never receive these reserved names. */
static inline int ggml_htp_output_weight_part(const char *name) {
    if (!name) return -1;
    const char *original = name;
    const char *separator = strchr(name, '#');
    if (separator) {
        size_t length = (size_t) (separator - name);
        if (!((length == 5 && memcmp(name, "MyHTP", 5) == 0) ||
              (length == 3 && memcmp(name, "CPU", 3) == 0))) return -1;
        original = separator + 1;
    }
    const char *names[] = { "htp.output.packed.0.weight", "htp.output.packed.1.weight" };
    for (int part = 0; part < 2; ++part) {
        size_t length = strlen(names[part]);
        if (strncmp(original, names[part], length) != 0) continue;
        const char *end = original + length;
        if (*end == '#') {
            ++end;
            if (*end < '0' || *end > '9') return -1;
            do { ++end; } while (*end >= '0' && *end <= '9');
        }
        if (*end == '\0') return part;
    }
    return -1;
}

/* Copy IEEE half bits from [vocabulary,K] into the existing HTP layout
 * [N/32,K/32,input-pair,output-column,pair-half]. No numerical conversion. */
static inline bool ggml_htp_pack_output_f16(uint16_t *packed, const uint16_t *rows,
                                           size_t k, size_t n) {
    if (!packed || !rows || !k || !n || k % 32 || n % 32 ||
        k > SIZE_MAX / n / sizeof(uint16_t)) return false;
    size_t index = 0;
    for (size_t nb = 0; nb < n; nb += 32)
        for (size_t kb = 0; kb < k; kb += 32)
            for (size_t pair = 0; pair < 16; ++pair)
                for (size_t lane = 0; lane < 32; ++lane)
                    for (size_t half = 0; half < 2; ++half)
                        packed[index++] = rows[(nb + lane) * k + kb + pair * 2 + half];
    return true;
}
