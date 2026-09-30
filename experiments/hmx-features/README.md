# Independent v68 HMX feature probes

See [the feature results](../../docs/HMX-FEATURES.md) for the V81-to-v68
compatibility table and [raw evidence](../../results/hmx-features-2026-09-29).
This standalone RPC module leaves the production inference backend unchanged.

Use a Linux build host with a separately installed Qualcomm SDK and Android
NDK, and an attached Snapdragon 888 phone with working unsigned cDSP FastRPC.
The tested configuration uses SDK 6.6.0.0, Tools 19.0.07 and NDK r26d.
Install CMake and Ninja on PATH; provide your own toolchain locations:

```sh
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.6.0.0
export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07"
export ANDROID_NDK=/path/to/android-ndk-r26d

python3 build.py
python3 compile.py
adb devices
python3 run.py --serial YOUR_ADB_SERIAL
```

The runner deploys only the Android executable, its RPC stub library and the
DSP skeleton to `/data/local/tmp/hmx-v68-features`. It uses firmware DSP
runtime libraries and never pushes SDK DSP C++ runtime libraries. Each case
runs sequentially in a fresh process with a 40-second device timeout and a
50-second host timeout. Logs are written under `logs/` and excluded from Git;
the checked-in results are a separately reviewed snapshot.

`run.py` records mismatches and continues through the case list. A zero exit
from the runner means collection completed, **not that every hypothesis
passed**. Inspect `logs/summary.json`: `passed=true` requires the specified
reference to pass; `passed=false` records a mismatch; `passed=null` reserves
judgment for an observation-only case. Every entry includes the exact probe
arguments and its process exit code. Raw logs retain all comparison counts,
raw half samples, unchanged-sentinel counts and resource/guard results.

For a selected reproduction:

```sh
python3 run.py --serial YOUR_ADB_SERIAL --case deep_32 --case weight_deep
python3 run.py --serial YOUR_ADB_SERIAL --skip-deploy --case baseline_final
```

Each invocation replaces `logs/summary.json` with that invocation's selected
cases. The runner compares the local and phone DSP SHA256, including with
`--skip-deploy`; a different deployed skeleton stops the run. Archive logs
before running another build. `compile.py` preserves separate diagnostics for
26 target/case combinations; the expected v68 modern-instruction rejections
are recorded without stopping the remaining compile probes.

`matrix_reference.py` and `epilogue_reference.py` supply independently designed
exact-fraction fixtures. They self-check on an ordinary Python host:

```sh
python3 matrix_reference.py
python3 epilogue_reference.py
```

The reference fixtures include modern operations whose implementation is
unavailable through the tested v68 API. Generating their references is not
evidence that the phone executed them. The actual phone corpus is defined in
`run.py`, with scalar comparisons in `host.c` and direct HMX instructions in
`dsp.c`.
