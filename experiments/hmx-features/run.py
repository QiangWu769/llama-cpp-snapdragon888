#!/usr/bin/env python3
"""Deploy an isolated probe and run each case in a fresh Android process."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

def spatial(x):
    return ((x >> 1) << 7) | ((x & 1) << 1)

def cases():
    result = []
    def add(name, feature=0, tiles=1, start=0, stop=31, pattern=1,
            act=None, store=2047, offset=0, mask=28, variant=0, bias=0,
            exploratory=False):
        if act is None:
            act = ((tiles - 1) << 11) | spatial(mask) | (stop << 2)
            if feature == 5:
                act = 2048 | spatial(mask) | (stop << 2)
        result.append(dict(name=name, args=[feature, tiles, start, stop, pattern,
                         act, store, offset, mask, variant, bias], exploratory=exploratory))
    for ctl in (0x7ff, 0x77c, 0x7fe):
        add(f'baseline_act_{ctl:x}', act=ctl)
        add(f'basic_act_{ctl:x}', feature=1, act=ctl)
    for pattern in (0, 2, 3):
        add(f'baseline_pattern_{pattern}', pattern=pattern, act=2047)
    for start, stop in ((0,7),(8,15),(16,31),(24,31),(0,15)):
        add(f'partial_{start}_{stop}', feature=2, start=start, stop=stop)
    for tiles in (2,3,8,32):
        add(f'deep_{tiles}', feature=3, tiles=tiles)
    add('deep_partial_3_24_15',feature=3,tiles=3,start=24,stop=15)
    add('deep_partial_2_16_7',feature=3,tiles=2,start=16,stop=7)
    add('weight_negate',feature=6)
    add('weight_negate_ones',feature=6,pattern=0)
    for feature,name in ((10,'accumulate'),(11,'retain'),(12,'convert_clear'),
                         (13,'explicit_clear'),(14,'repeat_retain'),(20,'explicit_swap')):
        add(name,feature=feature)
    for pattern in (0,1,3):
        add(f'positive_{pattern}',feature=15,pattern=pattern)
    add('before',feature=16,exploratory=True)
    add('weight_deep',feature=4)
    add('weight_deep_extra_swap_control',feature=4,variant=1,exploratory=True)
    for offset in (0,4,20,28):
        add(f'single_shift_{offset}',feature=5,offset=offset,pattern=5)
    add('single_negative_dy',feature=5,offset=20,pattern=5,variant=1)
    add('single_nonadjacent',feature=5,offset=20,pattern=1,variant=2)
    add('single_signed_full',feature=5,offset=20,pattern=1)
    add('single_partial',feature=5,offset=4,pattern=1,start=8,stop=15)
    for feature,name in ((17,'bias_legacy'),(18,'bias2')):
        for variant in (0,1):
            add(f'{name}_roundtrip_{variant}',feature=feature,bias=1,variant=variant)
    add('four_bias_banks_roundtrip',feature=28,bias=1)
    for bias in (2,3,4,5):
        add(f'custom_bias_{bias}',feature=19,bias=bias,pattern=3,exploratory=True)
    for pattern in (1,3):
        for bias in (4,7):
            for variant in (0,1):
                add(f'add_bias_p{pattern}_b{bias}_pos{variant}',feature=22,bias=bias,
                    pattern=pattern,variant=variant)
    for bias in (0,2,3,4,6):
        add(f'bias_amplitudes_{bias}',feature=25,bias=bias,pattern=6,exploratory=True)
    add('legacy_clip',feature=27,bias=2,pattern=6)
    for shape in range(8):
        add(f'v81_shape_{shape}_legacy_converter',feature=26,bias=8,pattern=1,variant=shape,exploratory=True)
    for bias,name in ((9,'input_bias'),(10,'scale'),(11,'output_bias'),(12,'combined')):
        add(f'v81_{name}_legacy_converter',feature=26,bias=bias,pattern=1,exploratory=True)
    add('clear_both_banks',feature=23)
    add('before_sequence',feature=24,exploratory=True)
    add('overflow',feature=21,pattern=4,exploratory=True)
    for variant in (1,2):
        add(f'overflow_sentinel_{variant}',feature=21,pattern=4,variant=variant,exploratory=True)
        add(f'before_sentinel_{variant}',feature=16,variant=variant,exploratory=True)
    add('overflow_positive64',feature=21,pattern=7,exploratory=True)
    add('overflow_negative64',feature=21,pattern=8,exploratory=True)
    add('baseline_final',act=2047)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',required=True)
    parser.add_argument('--phone-dir',default='/data/local/tmp/hmx-v68-features')
    parser.add_argument('--case',action='append',help='Run only matching named cases')
    parser.add_argument('--skip-deploy',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'/data/local/tmp/[A-Za-z0-9_-]+',args.phone_dir):
        parser.error('phone directory must be a simple /data/local/tmp directory')
    root=Path(__file__).resolve().parent
    logs=root/'logs';logs.mkdir(exist_ok=True)
    adb=['adb','-s',args.serial];phone=args.phone_dir
    def call(command):
        return subprocess.run(adb+command,stdin=subprocess.DEVNULL,check=True,capture_output=True,text=True,timeout=30)
    if not args.skip_deploy:
        call(['shell','-n','mkdir -p '+phone+'/dsp'])
        for source,target in ((root/'android_ReleaseG_aarch64/ship/hmx_probe_test',phone),
                              (root/'android_ReleaseG_aarch64/ship/libhmx_probe.so',phone),
                              (root/'hexagon_ReleaseG_toolv19_v68/ship/libhmx_probe_skel.so',phone+'/dsp')):
            call(['push',str(source),target+'/'+source.name])
        call(['shell','-n','chmod 755 '+phone+'/hmx_probe_test'])
    dsp=phone+'/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp;/system/lib/rfsa/adsp'
    binary=root/'hexagon_ReleaseG_toolv19_v68/ship/libhmx_probe_skel.so'
    local_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
    device_sha=call(['shell','-n','sha256sum '+phone+'/dsp/libhmx_probe_skel.so']).stdout.split()[0]
    if local_sha!=device_sha:
        raise SystemExit('deployed skeleton SHA256 differs from local build; deploy the matching build')
    (logs/'binary.json').write_text(json.dumps(dict(local_dsp_sha256=local_sha,phone_dsp_sha256=device_sha,
                                                  identical=True),indent=2)+'\n')
    summary=[]
    selected=cases()
    if args.case:
        selected=[case for case in selected if case['name'] in args.case]
        missing=set(args.case)-{case['name'] for case in selected}
        if missing:parser.error('unknown cases: '+str(sorted(missing)))
    for case in selected:
        command='cd '+phone+' && '+shlex.join(['env','LD_LIBRARY_PATH='+phone+':/vendor/lib64',
            'ADSP_LIBRARY_PATH='+dsp,'DSP_LIBRARY_PATH='+dsp,'timeout','40','./hmx_probe_test']+list(map(str,case['args'])))
        try:
            run=subprocess.run(adb+['shell','-n',command],stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=50)
            data=run.stdout+run.stderr;code=run.returncode
        except subprocess.TimeoutExpired as error:
            data=str(error);code=124
        (logs/('phone-'+case['name']+'.log')).write_text(data)
        numeric=[dict(output=int(o),checked=int(c),bad=int(b),nonfinite=int(n),max_abs_error=float(e))
                 for o,c,b,n,e in re.findall(r'NUMERIC output=(\d+) checked=(\d+) bad=(\d+) nonfinite=(\d+) max_abs_error=([^ ]+)',data)]
        observed=[dict(output=int(o),sentinel_equal=int(n),first_raw=raw)
                  for o,n,raw in re.findall(r'RAW output=(\d+) sentinel_equal=(\d+) data=([^\n]+)',data)]
        entry=dict(case,exit=code,numeric=numeric,observed=observed,rpc_completed='stage=130 result=0' in data,
                   reference_match=code==0 and 'FEATURE_TEST PASS' in data,
                   passed=None if case['exploratory'] else code==0 and 'FEATURE_TEST PASS' in data)
        summary.append(entry)
        print(json.dumps({k:v for k,v in entry.items() if k!='args'}),flush=True)
        (logs/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    main()
