#!/usr/bin/env python3
"""Build on Linux using a separately installed Hexagon SDK and Android NDK."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('hmx', 'hvx', 'cpu'), default='hmx')
    parser.add_argument('--sdk', default=os.environ.get('HEXAGON_SDK_ROOT'))
    parser.add_argument('--tools', default=os.environ.get('HEXAGON_TOOLS_ROOT'))
    parser.add_argument('--ndk', default=os.environ.get('ANDROID_NDK'))
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.ndk or (args.backend != 'cpu' and not args.sdk):
        parser.error('set ANDROID_NDK and, for HMX/HVX, HEXAGON_SDK_ROOT (or --ndk/--sdk)')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    root = Path(__file__).resolve().parents[1]
    ndk = Path(args.ndk).expanduser().resolve()
    env = os.environ.copy()
    logs = root / 'logs'

    def require(path):
        if not args.dry_run and not path.exists():
            parser.error('missing toolchain path: ' + str(path))

    def run(label, command):
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        logs.mkdir(exist_ok=True)
        log = logs / (label + '.log')
        with log.open('w') as output:
            result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        print(f'{label}: exit {result.returncode}; log: {log}', flush=True)
        if result.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-65:]))
            raise SystemExit(result.returncode)

    require(ndk / 'build/cmake/android.toolchain.cmake')
    if args.backend != 'cpu':
        sdk = Path(args.sdk).expanduser().resolve()
        compiler = Path(args.tools).expanduser().resolve() if args.tools else sdk / 'tools/HEXAGON_Tools/19.0.07'
        require(compiler)
        env.update(HEXAGON_SDK_ROOT=str(sdk), HEXAGON_TOOLS_ROOT=str(compiler))
        for arch in ('hexagon_ReleaseG_toolv19_v68', 'android_ReleaseG_aarch64'):
            android = arch.startswith('android')
            dest = root / 'htp-ops-lib' / (arch + ('_hmx' if args.backend == 'hmx' else ''))
            config = {
                'HEXAGON_SDK_ROOT': str(sdk), 'HEXAGON_CMAKE_ROOT': str(sdk / 'build/cmake'),
                'CMAKE_ARCHIVE_OUTPUT_DIRECTORY': str(dest / 'ship'),
                'CMAKE_LIBRARY_OUTPUT_DIRECTORY': str(dest / 'ship'),
                'CMAKE_RUNTIME_OUTPUT_DIRECTORY': str(dest / 'ship'), 'V': arch,
            }
            if android:
                config.update(ANDROID_ABI='arm64-v8a', ANDROID_NATIVE_API_LEVEL='26',
                              ANDROID_NDK=str(ndk), ANDROID_STL='none', CMAKE_SYSTEM_NAME='Android',
                              DSP_TYPE='3', OS_TYPE='HLOS', PREBUILT_LIB_DIR='android_aarch64')
            else:
                config.update(ADD_SYMBOLS='1', DO_SYSTEM_INCLUDE='0', DSP_VERSION='v68',
                              HEXAGON_TOOLS_ROOT=str(compiler), PREBUILT_LIB_DIR='hexagon_toolv19_v68',
                              QURT_OS='1')
            toolchain = sdk / 'build/cmake' / ('android_toolchain.cmake' if android else 'hexagon_toolchain.cmake')
            require(toolchain)
            config.update(CMAKE_TOOLCHAIN_FILE=str(toolchain), CMAKE_BUILD_TYPE='Release',
                          HTP_USE_HMX='ON' if args.backend == 'hmx' else 'OFF')
            run('configure-' + dest.name, ['cmake', '-S', str(root / 'htp-ops-lib'), '-B', str(dest),
                                         '-G', 'Ninja'] + [f'-D{k}={v}' for k, v in config.items()])
            run('build-' + dest.name, ['cmake', '--build', str(dest), '-j', str(args.jobs)])

    src = root / 'llama.cpp-npu'
    dest = src / ('build-android-cpu' if args.backend == 'cpu' else 'build-android')
    run('configure-' + dest.name, ['cmake', '-S', str(src), '-B', str(dest),
        '-DCMAKE_TOOLCHAIN_FILE=' + str(ndk / 'build/cmake/android.toolchain.cmake'),
        '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-26', '-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_HTP=' + ('OFF' if args.backend == 'cpu' else 'ON'), '-DGGML_OPENMP=OFF',
        '-DBUILD_SHARED_LIBS=ON', '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DGGML_NATIVE=OFF'])
    run('build-' + dest.name, ['cmake', '--build', str(dest), '-j', str(args.jobs), '--target',
        'llama-cli', 'llama-quantize', 'llama-bench', 'llama-perplexity'])


if __name__ == '__main__':
    main()
