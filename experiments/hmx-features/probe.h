#pragma once
#include <stdint.h>
#define PROBE_MAGIC 0x484d5846u
#define PROBE_BYTES 65536
#define MAX_OUTPUTS 8
enum feature {
    BASELINE, BASIC, PARTIAL, DEEP, WEIGHT_DEEP, SINGLE, WEIGHT_NEGATE,
    ACCUMULATE=10, RETAIN, CONVERT_CLEAR, EXPLICIT_CLEAR, REPEAT_RETAIN,
    POSITIVE, BEFORE, BIAS_ROUNDTRIP, BIAS2_ROUNDTRIP, CUSTOM_BIAS,
    EXPLICIT_SWAP, OVERFLOW, ADD_BIAS, CLEAR_BOTH, BEFORE_SEQUENCE, BIAS_OBSERVE,
    MODERN_BIAS_COMPAT, LEGACY_CLIP, BIAS_BANKS
};
struct probe_report {
    uint32_t magic;
    volatile int32_t stage;
    int32_t result;
    int32_t power_status, query_status, attr_status, hmx_attr_status, vtcm_attr_status;
    int32_t lock_status, unlock_status, release_status, power_down_status;
    uint32_t total_vtcm, available_vtcm, resource_id, vtcm_address;
    uint32_t has_lock1, has_lock2;
    uint32_t feature, tiles, start, stop, pattern, offset, spatial_mask;
    uint32_t act_control, store_control, variant, outputs, guard_bad;
    uint32_t untouched[MAX_OUTPUTS], raw[MAX_OUTPUTS][16];
    uint32_t bias_words[256], bias_readback[256];
    float output[MAX_OUTPUTS][1024];
};
static inline unsigned packed_index(unsigned row,unsigned col) {
    return (row/2)*64+col*2+row%2;
}
static inline unsigned spatial_bits(unsigned x) {
    return ((x>>1)<<7)|((x&1)<<1);
}
static inline unsigned output_sentinel(const struct probe_report *p) {
    if((p->feature==BEFORE||p->feature==OVERFLOW)&&p->variant==1)return 0x3555;
    if((p->feature==BEFORE||p->feature==OVERFLOW)&&p->variant==2)return 0xb955;
    return 0x7e00;
}
static inline float activation_value(unsigned t,unsigned r,unsigned k,unsigned pattern) {
    if(pattern==0)return 1;
    if(pattern==1)return (float)(((int)(t*3+r*2+k)%13)-6)/8;
    if(pattern==2)return r==k?1:0;
    if(pattern==3)return (float)((int)r-16)/8;
    if(pattern==4)return 256;
    if(pattern==6)return r==k?1:0;
    if(pattern==7)return 64;
    if(pattern==8)return -64;
    return (float)((int)((t*32+r)%37)-18)/8;
}
static inline float weight_value(unsigned t,unsigned k,unsigned n,unsigned group,unsigned pattern) {
    if(pattern==0)return group?2:1;
    if(pattern==1)return (float)(((int)(t*2+k+3*n+5*group)%11)-5)/8;
    if(pattern==2)return (float)(((int)(k+3*n+group)%19)-9)/8;
    if(pattern==3)return k==n?1:0;
    if(pattern==4)return 256;
    if(pattern==6) {
        const float values[16]={-2048,-64,-32,-4,-2,-1,-0.5,0,0.5,1,2,4,32,64,2048,65504};
        return values[n%16];
    }
    if(pattern==7||pattern==8)return 64;
    return k==n?1:0;
}
