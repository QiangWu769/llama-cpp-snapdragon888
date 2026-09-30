/* Compile-only instruction probes. This file does not acquire HMX or VTCM.
 * Compile each PROBE_KIND separately to preserve per-instruction diagnostics.
 * Source is independent of SDK headers; no SDK implementation is included.
 */
#include "operands.h"

#ifndef PROBE_KIND
#error "Define PROBE_KIND=0..5"
#endif

/* The caller supplies fully encoded scalar operands, all memory in VTCM. */
__attribute__((noinline))
void hmx_matrix_compile_probe(uint32_t act_rs, uint32_t act_rt,
                              uint32_t weight_rs, uint32_t weight_rt) {
#if PROBE_KIND == 0
    __asm__ volatile("{ activation.hf = mxmem(%0, %1)\n"
                     "  weight.hf = mxmem(%2, %3) }"
                     :: "r"(act_rs), "r"(act_rt),
                        "r"(weight_rs), "r"(weight_rt) : "memory");
#elif PROBE_KIND == 1
    __asm__ volatile("{ activation.hf = mxmem(%0, %1):deep\n"
                     "  weight.hf = mxmem(%2, %3) }"
                     :: "r"(act_rs), "r"(act_rt),
                        "r"(weight_rs), "r"(weight_rt) : "memory");
#elif PROBE_KIND == 2
    __asm__ volatile("{ activation.hf = mxmem(%0, %1)\n"
                     "  weight.hf = mxmem(%2, %3):deep }"
                     :: "r"(act_rs), "r"(act_rt),
                        "r"(weight_rs), "r"(weight_rt) : "memory");
#elif PROBE_KIND == 3
    __asm__ volatile("{ activation.hf = mxmem(%0, %1):single\n"
                     "  weight.hf = mxmem(%2, %3) }"
                     :: "r"(act_rs), "r"(act_rt),
                        "r"(weight_rs), "r"(weight_rt) : "memory");
#elif PROBE_KIND == 4
    (void)act_rs; (void)act_rt; (void)weight_rs; (void)weight_rt;
    __asm__ volatile("mxswapacc.hf" ::: "memory");
#elif PROBE_KIND == 5
    (void)act_rs; (void)act_rt; (void)weight_rs; (void)weight_rt;
    __asm__ volatile("mxclracc.hf" ::: "memory");
#else
#error "Unknown PROBE_KIND"
#endif
}
