#include <stdint.h>
#include <time.h>

static uint64_t ns(clockid_t id) {
    struct timespec value;
    if (clock_gettime(id, &value)) return UINT64_MAX;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

/* Read-only clock anchors. No PMU configuration or hardware state changes. */
int calibration_read_clocks(uint64_t *values, unsigned int count) {
    if (!values || count < 7) return -1;
    values[0] = ns(CLOCK_MONOTONIC);
    values[1] = ns(CLOCK_BOOTTIME);
    values[2] = ns(CLOCK_MONOTONIC_RAW);
    values[3] = ns(CLOCK_REALTIME);
#if defined(__aarch64__)
    __asm__ volatile("isb\n\tmrs %0, cntvct_el0\n\tmrs %1, cntfrq_el0"
                     : "=r"(values[4]), "=r"(values[5]) : : "memory");
#else
    values[4] = values[5] = 0;
#endif
    values[6] = ns(CLOCK_MONOTONIC);
    return 0;
}
