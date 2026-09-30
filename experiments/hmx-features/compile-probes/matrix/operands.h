#ifndef HMX_MATRIX_PROBE_OPERANDS_H
#define HMX_MATRIX_PROBE_OPERANDS_H

#include <stdint.h>

/* Candidate V81 field encodings. Their meaning on v68 requires device tests. */
static inline uint32_t matrix_spatial_bits(unsigned five_bits) {
    return ((five_bits >> 1) << 7) | ((five_bits & 1u) << 1);
}

static inline uint32_t matrix_activation_rs(uint32_t aligned_address,
                                             unsigned first_channel,
                                             unsigned spatial_offset) {
    return aligned_address | (first_channel << 2) |
           matrix_spatial_bits(spatial_offset);
}

static inline uint32_t matrix_activation_rt(unsigned last_channel,
                                             unsigned spatial_mask,
                                             uint32_t upper_field) {
    return upper_field | (last_channel << 2) |
           matrix_spatial_bits(spatial_mask);
}

static inline uint32_t matrix_deep_upper(unsigned croutons) {
    return (croutons - 1u) << 11;
}

/* A signed byte displacement, already a multiple of 2048. No signed shift. */
static inline uint32_t matrix_single_upper(int32_t second_tile_delta) {
    return (uint32_t)second_tile_delta;
}

static inline uint32_t matrix_weight_rt(unsigned selected_channels,
                                        unsigned filter_groups) {
    return 64u * selected_channels * filter_groups - 1u;
}

/* Index in uint16_t elements, not bytes. */
static inline unsigned matrix_activation_index(unsigned spatial,
                                                unsigned channel) {
    return (spatial / 2u) * 64u + 2u * channel + spatial % 2u;
}

static inline unsigned matrix_weight_index(unsigned channel,
                                            unsigned output_channel) {
    return (channel / 2u) * 64u + 2u * output_channel + channel % 2u;
}

#endif
