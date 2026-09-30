#pragma once
#include <stdint.h>

#define CALIBRATION_MAGIC UINT32_C(0x43414c38)
#define CALIBRATION_VERSION 1u
#define CALIBRATION_BUILD_UUID "a871b4db-ef2c-4a6f-8d33-f6836683123e"
#define CALIBRATION_REPORT_BYTES 16384u
#define CALIBRATION_MAX_PERIODS 128u
#define CALIBRATION_K 1024u
#define CALIBRATION_DIM 32u
#define CALIBRATION_VTCM_BYTES (256u * 1024u)
#define CALIBRATION_NOT_ATTEMPTED (-9999)
enum calibration_mode { CALIBRATION_SCALAR=0, CALIBRATION_HVX=1, CALIBRATION_HMX=2 };

struct calibration_period {
    uint64_t start_qtimer, end_qtimer, active_qtimer, elapsed_pcycles, active_pcycles, iterations;
};
struct calibration_report {
    uint32_t magic, version, report_size, mode;
    uint32_t duration_us, period_us, duty_percent, chunk_iterations, workers;
    volatile int32_t stage;
    int32_t result, power_vote_status, power_up_status, lock_status, unlock_status;
    int32_t release_status, power_down_status, dcvs_reset_status, sleep_status;
    uint32_t period_count, vtcm_bytes, total_vtcm, available_vtcm;
    uint64_t measurement_start_qtimer, measurement_end_qtimer;
    uint64_t measurement_start_pcycles, measurement_end_pcycles;
    uint64_t active_qtimer, active_pcycles, active_iterations, warmup_iterations;
    uint32_t scalar_values[4];
    uint32_t hvx_values[128];
    float hmx_values[1024];
    struct calibration_period periods[CALIBRATION_MAX_PERIODS];
};

static inline uint32_t calibration_initial(unsigned i) { return UINT32_C(0x01234567) + i * UINT32_C(0x10203); }
static inline uint32_t calibration_increment(unsigned i) { return UINT32_C(0x13579bdf) + (i % 4u) * UINT32_C(0x2468ace); }
static inline uint32_t calibration_expected(unsigned i, uint64_t iterations) {
    return calibration_initial(i) + (uint32_t)iterations * calibration_increment(i / 32u);
}
static inline float calibration_hmx_expected(unsigned row, unsigned column) {
    return (float)(row+1u) * ((int)(column%7u)-3) / 4.0f;
}
