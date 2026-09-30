/* Independent native IEEE conversion oracle for the v68 integer helpers. */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "v68_fp_bits.h"

static void check_float(uint32_t bits) {
  float input;
  __fp16 reference;
  uint16_t expected;
  memcpy(&input, &bits, sizeof(input));
  reference = (__fp16) input;
  memcpy(&expected, &reference, sizeof(expected));
  const uint16_t got = hvx_v68_sf_bits_to_hf(bits);
  if (isnan(input)) {
    assert((got & 0x7c00u) == 0x7c00u && (got & 0x3ffu) != 0);
    assert((got & 0x8000u) == (expected & 0x8000u));
  } else if (got != expected) {
    fprintf(stderr, "float bits %08x: got half %04x, expected %04x\n", bits, got, expected);
    assert(0);
  }
}

int main(void) {
  for (uint32_t h = 0; h <= 65535u; ++h) {
    const uint16_t bits = (uint16_t) h;
    __fp16 input;
    float reference;
    uint32_t expected;
    memcpy(&input, &bits, sizeof(input));
    reference = (float) input;
    memcpy(&expected, &reference, sizeof(expected));
    const uint32_t got = hvx_v68_hf_bits_to_sf(bits);
    if (isnan(reference)) {
      assert((got & 0x7f800000u) == 0x7f800000u && (got & 0x7fffffu) != 0);
      assert((got & 0x80000000u) == (expected & 0x80000000u));
    } else assert(got == expected);
    check_float(expected);
    /* Values immediately around each representable half and rounding midpoint. */
    if ((h & 0x7c00u) != 0x7c00u) {
      check_float(expected + 1u);
      if (expected != 0) check_float(expected - 1u);
      if (h < 0x7bffu) {
        uint16_t next_bits = (uint16_t) (h + 1u);
        __fp16 next_half;
        memcpy(&next_half, &next_bits, sizeof(next_half));
        float midpoint = (reference + (float) next_half) * 0.5f;
        uint32_t midpoint_bits;
        memcpy(&midpoint_bits, &midpoint, sizeof(midpoint_bits));
        check_float(midpoint_bits);
        check_float(midpoint_bits + 1u);
        if (midpoint_bits != 0) check_float(midpoint_bits - 1u);
      }
    }
  }
  uint32_t random = 1;
  for (int i = 0; i < 1000000; ++i) {
    random = random * 1664525u + 1013904223u;
    check_float(random);
  }
  puts("PASS all 65536 binary16 patterns, rounding boundaries and 1000000 binary32 bit patterns");
  return 0;
}
