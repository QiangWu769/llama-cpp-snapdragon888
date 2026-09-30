# CPU/HTP partitioning regression

Run `./experiments/utilization/split/run.sh` from the repository root. The test
builds the normal CPU library on the host and exercises the actual shared
capability helper and CPU device support function with UndefinedBehaviorSanitizer.
It does not create a DSP session or require a phone, model or Qualcomm SDK.

The fixtures cover buffer-free loader probes, packed F16/Q4_0/Q8_0/IQ4_NL
matrices, M1 dimensions, unsupported strides/shapes/types, the two runtime output
cache names, blocked CPU fallback, ordinary CPU operations and exact flash
attention layout/scale/mask restrictions. The toggle is off by default and only
its exact value `1` enables the experiment.

Optional arguments select the source tree and a retained build directory:
`./experiments/utilization/split/run.sh /path/to/llama.cpp-npu /tmp/split-build`.
Physical inference accuracy and performance are separate validation steps.
