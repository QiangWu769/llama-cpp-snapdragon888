#!/usr/bin/env python3
"""Deploy built artifacts to a separate adb-accessible phone directory."""
import argparse
import os
import posixpath
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('hmx', 'hvx', 'cpu'), default='hmx')
    parser.add_argument('--serial', default=os.environ.get('ANDROID_SERIAL'))
    parser.add_argument('--phone-dir')
    parser.add_argument('--ndk', default=os.environ.get('ANDROID_NDK'))
    parser.add_argument('--model', type=Path, help='Explicit GGUF to copy; HTP and CPU need different layouts')
    parser.add_argument('--dry-run', action='store_true', help='Check local artifacts and print adb commands')
    args = parser.parse_args()
    if not args.ndk:
        parser.error('set ANDROID_NDK or --ndk for the ARM64 libc++_shared.so')
    root = Path(__file__).resolve().parents[1]
    phone = posixpath.normpath(args.phone_dir or '/data/local/tmp/llama-v68-' + args.backend).rstrip('/')
    if not phone.startswith('/') or not phone or ':' in phone or ';' in phone:
        parser.error('--phone-dir must be an absolute directory without : or ;')
    adb = ['adb'] + (['-s', args.serial] if args.serial else [])
    build = root / 'llama.cpp-npu' / ('build-android-cpu' if args.backend == 'cpu' else 'build-android')
    files = [build / 'bin' / name for name in ('llama-cli', 'llama-quantize', 'llama-bench', 'llama-perplexity')]
    libraries = sorted(build.rglob('*.so'))
    expected = {'libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so'}
    if args.backend != 'cpu':
        expected.add('libggml-htp.so')
    missing = expected - {path.name for path in libraries}
    if missing:
        parser.error('build first; missing libraries: ' + ', '.join(sorted(missing)))
    files.extend(libraries)
    runtime = sorted(Path(args.ndk).expanduser().glob(
        'toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'))
    if len(runtime) != 1:
        parser.error('expected one ARM64 libc++_shared.so in the supplied NDK')
    files.extend(runtime)
    dsp = None
    if args.backend != 'cpu':
        suffix = '_hmx' if args.backend == 'hmx' else ''
        ops = root / 'htp-ops-lib'
        host = ops / ('android_ReleaseG_aarch64' + suffix) / 'ship'
        files.extend([host / 'libhtp_ops.so', host / 'htp_v68_test'])
        dsp = ops / ('hexagon_ReleaseG_toolv19_v68' + suffix) / 'ship/libhtp_ops_skel.so'
    if args.model:
        if args.model.suffix.lower() != '.gguf':
            parser.error('--model must name a .gguf file')
        files.append(args.model.expanduser().resolve())
    for path in files + ([dsp] if dsp else []):
        if not path.is_file():
            parser.error('missing artifact: ' + str(path))
    # Flatten shared objects only after verifying no ambiguous basename collision.
    unique = {}
    for path in files:
        if path.name in unique and unique[path.name].resolve() != path.resolve():
            parser.error('duplicate deployment basename: ' + path.name)
        unique[path.name] = path

    def run(*command):
        argv = adb + list(command)
        print(shlex.join(argv), flush=True)
        if not args.dry_run:
            subprocess.run(argv, check=True, stdin=subprocess.DEVNULL)

    run('shell', '-n', 'mkdir -p ' + shlex.quote(phone + '/dsp'))
    for name, path in unique.items():
        run('push', str(path), phone + '/' + name)
    if dsp:
        # Firmware DSP libc++/libc++abi are required. SDK 19 runtimes depend on
        # aligned_alloc, which the tested Snapdragon 888 firmware does not export.
        run('push', str(dsp), phone + '/dsp/' + dsp.name)
    executables = ['llama-cli', 'llama-quantize', 'llama-bench', 'llama-perplexity']
    if dsp:
        executables.append('htp_v68_test')
    run('shell', '-n', shlex.join(['chmod', '755'] + [phone + '/' + name for name in executables]))


if __name__ == '__main__':
    main()
