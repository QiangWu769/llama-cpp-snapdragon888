#pragma once

#include "dsp/hvx_internal.h"

#if HTP_HMX_V68
/* qf16 has fewer significant bits than IEEE binary16. A conversion through
 * qf16 (+0) therefore loses precision even when the source is already FP16.
 * Use integer bit conversions on v68; retain round-to-nearest, ties-to-even,
 * signed zero, subnormals and infinities. NaNs are quieted with their payload. */
static HVX_INLINE_ALWAYS uint32_t hvx_v68_hf_bits_to_sf(uint16_t bits) {
  const uint32_t sign = (uint32_t) (bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 31u;
  uint32_t mantissa = bits & 1023u;
  if (exponent == 31u) {
    return sign | 0x7f800000u | (mantissa << 13) | (mantissa ? 0x400000u : 0);
  }
  if (exponent == 0) {
    if (mantissa == 0) return sign;
    exponent = 113;
    while ((mantissa & 1024u) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    return sign | (exponent << 23) | ((mantissa & 1023u) << 13);
  }
  return sign | ((exponent + 112u) << 23) | (mantissa << 13);
}

static HVX_INLINE_ALWAYS uint16_t hvx_v68_sf_bits_to_hf(uint32_t bits) {
  const uint16_t sign = (uint16_t) ((bits >> 16) & 0x8000u);
  const uint32_t exponent = (bits >> 23) & 255u;
  const uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 255u) {
    return (uint16_t) (sign | 0x7c00u | (mantissa ? ((mantissa >> 13) | 0x200u) : 0));
  }
  const int unbiased = (int) exponent - 127;
  if (unbiased > 15) return (uint16_t) (sign | 0x7c00u);
  if (unbiased < -25) return sign;
  if (unbiased < -14) {
    const uint32_t value = mantissa | 0x800000u;
    const int shift = -unbiased - 1;
    uint32_t rounded = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1u));
    return (uint16_t) (sign | rounded);
  }
  uint32_t rounded = ((uint32_t) (unbiased + 15) << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  rounded += remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u));
  return (uint16_t) (sign | rounded);
}
#endif

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_wsf_to_vhf(HVX_Vector v1, HVX_Vector v0) {
#if HTP_HMX_V68
  const HVX_Vector sign_mask = Q6_V_vsplat_R(0x8000);
  const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fffffff);
  const HVX_Vector one = Q6_V_vsplat_R(1);
  const HVX_Vector round_bias = Q6_V_vsplat_R(0xfff);
  const HVX_Vector exponent_bias = Q6_V_vsplat_R(112 << 10);
  HVX_Vector a = Q6_V_vand_VV(v0, abs_mask);
  HVX_Vector b = Q6_V_vand_VV(v1, abs_mask);
  const HVX_Vector sign_a = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v0, 16), sign_mask);
  const HVX_Vector sign_b = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(v1, 16), sign_mask);
  /* Adding 0xfff plus the retained LSB implements nearest, ties-to-even. */
  a = Q6_Vw_vadd_VwVw(a, Q6_Vw_vadd_VwVw(round_bias,
      Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 13), one)));
  b = Q6_Vw_vadd_VwVw(b, Q6_Vw_vadd_VwVw(round_bias,
      Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(b, 13), one)));
  a = Q6_V_vor_VV(sign_a, Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(a, 13), exponent_bias));
  b = Q6_V_vor_VV(sign_b, Q6_Vw_vsub_VwVw(Q6_Vuw_vlsr_VuwR(b, 13), exponent_bias));
  const HVX_Vector packed = Q6_V_vor_VV(Q6_V_vand_VV(a, Q6_V_vsplat_R(0xffff)),
                                      Q6_Vw_vasl_VwR(b, 16));
  __attribute__((aligned(128))) uint32_t low[32], high[32];
  __attribute__((aligned(128))) uint16_t half[64];
  vmem(low) = v0;
  vmem(high) = v1;
  vmem(half) = packed;
  /* Only underflow, overflow and nonfinite lanes leave the vector fast path. */
  for (int i = 0; i < 32; ++i) {
    const uint32_t lo_abs = low[i] & 0x7fffffffu;
    const uint32_t hi_abs = high[i] & 0x7fffffffu;
    if (lo_abs < 0x38800000u || lo_abs >= 0x47800000u)
      half[2 * i] = hvx_v68_sf_bits_to_hf(low[i]);
    if (hi_abs < 0x38800000u || hi_abs >= 0x47800000u)
      half[2 * i + 1] = hvx_v68_sf_bits_to_hf(high[i]);
  }
  return vmem(half);
#else
  const HVX_Vector v_zero = Q6_V_vzero();

  HVX_Vector v0_qf32 = Q6_Vqf32_vadd_VsfVsf(v0, v_zero);
  HVX_Vector v1_qf32 = Q6_Vqf32_vadd_VsfVsf(v1, v_zero);

  return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(v1_qf32, v0_qf32));
#endif
}

static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vqf16_to_wqf32(HVX_Vector v_src) {
  const HVX_Vector v_lo_mask = Q6_V_vsplat_R(0x0000ffff);
  const HVX_Vector v_hi_mask = Q6_V_vsplat_R(0xffff0000);
  const HVX_Vector v_shift16 = Q6_V_vsplat_R(16);

  // adapted from qhmath_hvx_vqf32_convert_vqf16 (in qhmath_hvx_convert.h)
  // extract packed exp & mantissa
  HVX_Vector exp_comp = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0x1f));    // exp component: low 5 bits
  HVX_Vector mantissa = Q6_V_vand_VV(v_src, Q6_Vh_vsplat_R(0xffe0));  // mantissa: bits 5~15

  // Convert qf16 biased exponent to qf32 biased exponent
  // new exp = exp + ( 127 (qf32 bias) -15(qf16 bias) ) = 112
  exp_comp = Q6_Vh_vadd_VhVh(exp_comp, Q6_Vh_vsplat_R(112));

  // elements index in v_src: [0, n, 1, n+1, ..., 31, n+31]
  // unpack into [0, 1, ..., 31], [n, n+1, ..., n+31]

  // unpack exp
  HVX_Vector exp_comp0 = Q6_V_vand_VV(exp_comp, v_lo_mask);  // keep low 16 bits
  HVX_Vector exp_comp1 = Q6_Vw_vlsr_VwVw(exp_comp, v_shift16);

  // unpack mantissa + convert qf16 mantissa to qf32 mantissa (left shift 16 bits)
  HVX_Vector mantissa0 = Q6_Vw_vasl_VwVw(mantissa, v_shift16);
  HVX_Vector mantissa1 = Q6_V_vand_VV(mantissa, v_hi_mask);  // keep high 16 bits

  // merge qf32 exp + mantissa
  HVX_Vector v0_qf32 = Q6_Vw_vadd_VwVw(mantissa0, exp_comp0);
  HVX_Vector v1_qf32 = Q6_Vw_vadd_VwVw(mantissa1, exp_comp1);

  return Q6_W_vcombine_VV(v1_qf32, v0_qf32);
}

static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vqf16_to_wsf(HVX_Vector v_src) {
  HVX_VectorPair vp = hvx_my_vqf16_to_wqf32(v_src);

  HVX_Vector v0_sf = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vp));
  HVX_Vector v1_sf = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vp));
  return Q6_W_vcombine_VV(v1_sf, v0_sf);
}

static HVX_INLINE_ALWAYS HVX_Vector hvx_my_vhf_to_vqf16(HVX_Vector vx) {
  // converts fp16 to qf16 (using *1 or +0?)
  // return Q6_Vqf16_vmpy_VhfVhf(vx, Q6_Vh_vsplat_R(0x3c00));
  return Q6_Vqf16_vadd_VhfVhf(vx, Q6_V_vzero());
}

#if HTP_HMX_V68
/* Input contains one zero-extended half in each word. All 65536 patterns are
 * handled by integer HVX, including normalization of binary16 subnormals. */
static HVX_INLINE_ALWAYS HVX_Vector hvx_v68_halfwords_to_sf(HVX_Vector halves) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector mantissa_mask = Q6_V_vsplat_R(0x3ff);
  const HVX_Vector exponent_mask = Q6_V_vsplat_R(0x7c00);
  const HVX_Vector mantissa = Q6_V_vand_VV(halves, mantissa_mask);
  const HVX_Vector exponent = Q6_V_vand_VV(halves, exponent_mask);
  const HVX_Vector sign = Q6_Vw_vasl_VwR(Q6_V_vand_VV(halves, Q6_V_vsplat_R(0x8000)), 16);
  HVX_Vector normal = Q6_Vw_vadd_VwVw(
      Q6_Vw_vasl_VwR(Q6_V_vand_VV(halves, Q6_V_vsplat_R(0x7fff)), 13),
      Q6_V_vsplat_R(112 << 23));
  const HVX_Vector shift = Q6_Vw_vsub_VwVw(Q6_Vuw_vcl0_Vuw(mantissa), Q6_V_vsplat_R(21));
  HVX_Vector subnormal = Q6_V_vor_VV(
      Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(113), shift), 23),
      Q6_Vw_vasl_VwR(Q6_V_vand_VV(Q6_Vw_vasl_VwVw(mantissa, shift), mantissa_mask), 13));
  const HVX_VectorPred is_zero_mantissa = Q6_Q_vcmp_eq_VwVw(mantissa, zero);
  subnormal = Q6_V_vmux_QVV(is_zero_mantissa, zero, subnormal);
  HVX_Vector special = Q6_V_vor_VV(Q6_Vw_vasl_VwR(mantissa, 13), Q6_V_vsplat_R(0x7fc00000));
  special = Q6_V_vmux_QVV(is_zero_mantissa, Q6_V_vsplat_R(0x7f800000), special);
  normal = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(exponent, zero), subnormal, normal);
  normal = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(exponent, exponent_mask), special, normal);
  return Q6_V_vor_VV(sign, normal);
}
#endif

static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vhf_to_wsf(HVX_Vector vx) {
#if HTP_HMX_V68
  const HVX_Vector low = Q6_V_vand_VV(vx, Q6_V_vsplat_R(0xffff));
  const HVX_Vector high = Q6_Vuw_vlsr_VuwR(vx, 16);
  return Q6_W_vcombine_VV(hvx_v68_halfwords_to_sf(high), hvx_v68_halfwords_to_sf(low));
#else
  HVX_Vector v_src = hvx_my_vhf_to_vqf16(vx);
  return hvx_my_vqf16_to_wsf(v_src);
#endif
}

static HVX_INLINE_ALWAYS HVX_VectorPair hvx_my_vhf_to_wqf32(HVX_Vector vx) {
  HVX_Vector v_src = hvx_my_vhf_to_vqf16(vx);
  return hvx_my_vqf16_to_wqf32(v_src);
}
