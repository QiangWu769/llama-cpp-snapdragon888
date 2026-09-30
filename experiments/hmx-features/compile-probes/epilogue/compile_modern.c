/* Negative/positive architecture-gate probe, not a runtime entry point.
 * Compile CASE 0 and 1 separately with -mv68, then -mv73 and -mv81.
 * SDK19 previously rejected both instructions for -mv68. Do not override
 * the architecture guard or inject raw instruction encodings on that basis.
 */
#ifndef PROBE_CASE
#define PROBE_CASE 0
#endif
void hmx_epilogue_modern_probe(void *out, unsigned control) {
    (void)out;
    (void)control;
#if PROBE_CASE == 0
    __asm__ volatile("cvt.hf = acc(%0)" :: "r"(control) : "memory");
#elif PROBE_CASE == 1
    __asm__ volatile("mxmem(%0,%1) = cvt" :: "r"(out), "r"(control) : "memory");
#else
#error PROBE_CASE must be 0 or 1
#endif
}
