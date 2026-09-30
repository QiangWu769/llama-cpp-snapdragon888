#!/usr/bin/env python3
"""Build on the Ubuntu host with its installed SDK; no phone changes."""
from pathlib import Path
import os, subprocess

src=Path(__file__).resolve().parent
sdk=Path(os.environ['HEXAGON_SDK_ROOT'])
ndk=Path(os.environ['ANDROID_NDK'])
env=os.environ.copy()
env['HEXAGON_SDK_ROOT']=str(sdk)
env['HEXAGON_TOOLS_ROOT']=os.environ.get('HEXAGON_TOOLS_ROOT',str(sdk/'tools/HEXAGON_Tools/19.0.07'))
logs=src/'logs';logs.mkdir(exist_ok=True)
for arch in ['hexagon_ReleaseG_toolv19_v68','android_ReleaseG_aarch64']:
    dest=src/arch
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
    commands=[('configure',['cmake','-S',str(src),'-B',str(dest),'-G','Ninja']+['-D'+k+'='+v for k,v in config.items()]),
              ('build',['cmake','--build',str(dest),'-j','4'])]
    for label,command in commands:
        log=logs/(arch+'-'+label+'.log')
        with log.open('w') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        print(arch,label,'exit',r.returncode,flush=True)
        if r.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-70:]));raise SystemExit(r.returncode)
