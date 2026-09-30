#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "calibration.h"
int main(void) {
    unsigned checked=0;
    assert(sizeof(struct calibration_report)<CALIBRATION_REPORT_BYTES);
    assert(offsetof(struct calibration_report,measurement_start_qtimer)%8==0);
    const uint64_t counts[]={0,1,2,65536,UINT64_C(0xffffffff),UINT64_C(0x100000000),UINT64_C(0x100000001)};
    for(unsigned i=0;i<128;++i)for(unsigned c=0;c<sizeof(counts)/sizeof(counts[0]);++c) {
        uint64_t ref=UINT64_C(0x01234567)+(uint64_t)i*UINT64_C(0x10203)+
            counts[c]*(UINT64_C(0x13579bdf)+(uint64_t)(i/32)*UINT64_C(0x2468ace));
        assert(calibration_expected(i,counts[c])==(uint32_t)ref);++checked;
    }
    for(unsigned row=0;row<32;++row)for(unsigned col=0;col<32;++col) {
        double a=((double)row+1)/1024.0,b=((int)(col%7)-3)/4.0,sum=0;
        for(unsigned k=0;k<1024;++k)sum+=a*b;
        assert(calibration_hmx_expected(row,col)==sum);++checked;
    }
    assert(calibration_hmx_expected(0,0)==-0.75f);
    assert(calibration_hmx_expected(31,6)==24.0f);
    printf("ORACLE_PASS cases=%u report_size=%zu timer_offset=%zu\n",checked,sizeof(struct calibration_report),offsetof(struct calibration_report,measurement_start_qtimer));
}
