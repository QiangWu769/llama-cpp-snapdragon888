/* Standalone compile probe, not an RPC entry point or a runtime test.
 * Compile each case separately with the installed compiler's -mv68 -mhmx.
 * Example: hexagon-clang -mv68 -mhmx -O2 -c -DPROBE_CASE=1 compile_legacy.c
 * The caller supplies aligned VTCM pointers, owns HMX on this thread, and
 * validates every output. This file deliberately performs no resource setup.
 * Syntax matches the installed SDK header; acceptance is not hardware proof.
 */
#ifndef PROBE_CASE
#define PROBE_CASE 0
#endif

void hmx_epilogue_legacy_probe(void *out, const void *source, unsigned control) {
    (void)out;
    (void)source;
    (void)control;
    __asm__ volatile("barrier" ::: "memory");
#if PROBE_CASE == 0
    __asm__ volatile("mxmem(%0,%1):after.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 1
    __asm__ volatile("mxmem(%0,%1):after:retain.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 2
    __asm__ volatile("mxmem(%0,%1):after:pos.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 3
    __asm__ volatile("mxmem(%0,%1):after:retain:pos.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 4
    __asm__ volatile("mxmem(%0,%1):before.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 5
    __asm__ volatile("mxmem(%0,%1):before:retain.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 6
    __asm__ volatile("mxmem(%0,%1):before:pos.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 7
    __asm__ volatile("mxmem(%0,%1):before:retain:pos.hf = acc" :: "r"(out), "r"(control) : "memory");
#elif PROBE_CASE == 8
    __asm__ volatile("mxclracc.hf" ::: "memory");
#elif PROBE_CASE == 9
    __asm__ volatile("mxswapacc.hf" ::: "memory");
#elif PROBE_CASE == 10
    __asm__ volatile("bias = mxmem(%0)" :: "r"(source) : "memory");
#elif PROBE_CASE == 11
    __asm__ volatile("mxmem(%0) = bias" :: "r"(out) : "memory");
#elif PROBE_CASE == 12
    __asm__ volatile("bias = mxmem2(%0)" :: "r"(source) : "memory");
#elif PROBE_CASE == 13
    __asm__ volatile("mxmem2(%0) = bias" :: "r"(out) : "memory");
#else
#error PROBE_CASE must be in the range 0..13
#endif
    __asm__ volatile("barrier" ::: "memory");
}
