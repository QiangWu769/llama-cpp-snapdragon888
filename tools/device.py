#!/usr/bin/env python3
"""Run a bounded command on the phone with the correct library search paths."""
import argparse
import os
import posixpath
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('hmx', 'hvx', 'cpu'), default='hmx')
    parser.add_argument('--serial', default=os.environ.get('ANDROID_SERIAL'))
    parser.add_argument('--phone-dir')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='Use -- followed by the device executable and arguments')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout < 1:
        parser.error('supply a command after -- and a positive timeout')
    phone = posixpath.normpath(args.phone_dir or '/data/local/tmp/llama-v68-' + args.backend).rstrip('/')
    if not phone.startswith('/') or not phone or ':' in phone or ';' in phone:
        parser.error('--phone-dir must be an absolute directory without : or ;')
    env = {'LD_LIBRARY_PATH': phone + ':/vendor/lib64', 'HTP_TRACE': '1' if args.trace else '0'}
    if args.backend != 'cpu':
        dsp = phone + '/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp;/system/lib/rfsa/adsp'
        env.update(ADSP_LIBRARY_PATH=dsp, DSP_LIBRARY_PATH=dsp)
    remote = 'cd ' + shlex.quote(phone) + ' && ' + shlex.join(
        ['env'] + [k + '=' + v for k, v in env.items()] + ['timeout', str(args.timeout)] + command)
    adb = ['adb'] + (['-s', args.serial] if args.serial else [])
    argv = adb + ['shell', '-n', remote]
    # Keep stdout suitable for llama-bench JSON capture.
    import sys
    print(shlex.join(argv), file=sys.stderr, flush=True)
    if not args.dry_run:
        try:
            raise SystemExit(subprocess.run(argv, stdin=subprocess.DEVNULL, timeout=args.timeout + 15).returncode)
        except subprocess.TimeoutExpired:
            parser.exit(124, 'adb timed out; inspect the phone process if its connection was lost\n')


if __name__ == '__main__':
    main()
