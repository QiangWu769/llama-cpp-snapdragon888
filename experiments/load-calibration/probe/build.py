"""Build an isolated calibration payload with an externally installed SDK; no device calls."""
import argparse
import os
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sdk',required=True,type=Path)
parser.add_argument('--ndk',required=True,type=Path)
parser.add_argument('--qaic-bin',type=Path)
options=parser.parse_args()
source=Path(__file__).resolve().parent
sdk=options.sdk.resolve();ndk=options.ndk.resolve()
env=os.environ.copy();env['HEXAGON_SDK_ROOT']=str(sdk)
env['HEXAGON_TOOLS_ROOT']=str(sdk/'tools/HEXAGON_Tools/19.0.07')
if options.qaic_bin:env['PATH']=str(options.qaic_bin.resolve())+os.pathsep+env['PATH']
logs=source/'logs';logs.mkdir(exist_ok=True)
for arch in ['hexagon_ReleaseG_toolv19_v68','android_ReleaseG_aarch64']:
    dest=source/arch
    config={'HEXAGON_SDK_ROOT':str(sdk),'HEXAGON_CMAKE_ROOT':str(sdk/'build/cmake'),
            'CMAKE_ARCHIVE_OUTPUT_DIRECTORY':str(dest/'ship'),'CMAKE_LIBRARY_OUTPUT_DIRECTORY':str(dest/'ship'),
            'CMAKE_RUNTIME_OUTPUT_DIRECTORY':str(dest/'ship'),'V':arch,'CMAKE_BUILD_TYPE':'Release'}
    if arch.startswith('android'):
        config.update(ANDROID_ABI='arm64-v8a',ANDROID_NATIVE_API_LEVEL='26',ANDROID_NDK=str(ndk),
                      ANDROID_STL='none',CMAKE_SYSTEM_NAME='Android',DSP_TYPE='3',OS_TYPE='HLOS',
                      PREBUILT_LIB_DIR='android_aarch64',CMAKE_TOOLCHAIN_FILE=str(sdk/'build/cmake/android_toolchain.cmake'))
    else:
        config.update(ADD_SYMBOLS='1',DO_SYSTEM_INCLUDE='0',DSP_VERSION='v68',
                      HEXAGON_TOOLS_ROOT=env['HEXAGON_TOOLS_ROOT'],PREBUILT_LIB_DIR='hexagon_toolv19_v68',
                      QURT_OS='1',CMAKE_TOOLCHAIN_FILE=str(sdk/'build/cmake/hexagon_toolchain.cmake'))
    for label,command in [('configure',['cmake','-S',str(source),'-B',str(dest),'-G','Ninja']+
                          ['-D'+k+'='+v for k,v in config.items()]),
                          ('build',['cmake','--build',str(dest),'-j','4'])]:
        log=logs/(arch+'-'+label+'.log')
        with log.open('w') as f:
            result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        print(arch,label,'exit',result.returncode,flush=True)
        if result.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-70:]));raise SystemExit(result.returncode)
