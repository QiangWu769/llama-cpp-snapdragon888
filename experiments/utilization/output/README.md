# Experimental packed output projection tests

`HTP_OFFLOAD_OUTPUT=1` enables a runtime-owned packed output cache for the
validated tied Qwen2 F16 `[896,151936]` model. The embedding and original output
tensor retain their ordinary layout. The cache contains two vocabulary halves
of 75,968 columns, totaling 272,269,312 additional bytes. The default is off.
Unsupported storage, shape, architecture, missing HTP layer buffers, and active
LoRA adapters fail visibly.

These tests cover packing and lifetime properties. Numerical comparisons to
ordinary CPU inference and performance measurements are documented separately
in [UTILIZATION.md](../../../docs/UTILIZATION.md). Bit-exact packing does not
guarantee identical CPU logits: the existing HMX path rounds activations and
outputs to FP16.

Run the dependency-free host regressions from any directory:

```sh
sh experiments/utilization/output/run-regressions.sh
```

The runner uses a portable temporary directory, cleans its binaries on exit,
and enables UBSan. Set `CC` and `CXX` if needed. It runs:

- An independent inverse packing oracle, both full Qwen2 vocabulary halves,
  asymmetric tiles, all half encodings, input/output guards, invalid/overflow
  dimensions and exact/malformed cache names.
- The production DSP mapping manager with test-owned HAP function substitutes:
  a failed put retains the mapping, and a successful retry releases it.
- The production host mapper with test-owned RPCMEM/FastRPC substitutes:
  active and pending retirement, pending-unmap failure retention, retry,
  idempotence and mapping reuse.

The full-sized packing fixture needs roughly 260 MiB of transient host memory.
No model weights, SDK implementation, phone or server are required.

To additionally execute the actual host GGML concat and backend-buffer mapper
tests, first build matching host GGML libraries and supply their build tree:

```sh
GGML_HOST_BUILD=/path/to/matching/host-build \
  sh experiments/utilization/output/run-regressions.sh
```

The concat test checks each token's ordered vocabulary halves for M=1 and
M=32, including unequal halves. The extra mapper test uses real GGML buffer
objects and checks LRU capacity accounting after retirement. The runner reports
these optional tests as skipped when no host build is supplied.

The `mock/HAP_mem.h` file is a test-owned substitute containing only declarations
for the two functions used by the mapping manager. It is not an SDK header.
The [reload caller](reload/README.md) performs physical same-process lifecycle
validation through the llama library API.
