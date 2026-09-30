# Fast vocabulary concatenation regression

From the repository root, run `./experiments/utilization/concat/run.sh`.
No phone, model or Qualcomm SDK is required. Optional arguments select another
source tree and a retained build directory.

The regression executes 150 graph cases through both actual C executors:
normal CPU and HTP hybrid with DSP calls replaced by test stubs. Cases cover
F32/I32, M1/M32, the complete two-part vocabulary shape, asymmetric parts,
padding and permuted higher axes, scalar strides, concat dimensions1/2/3,
source/destination overlap fallback, exact bits, untouched source data and
padding/guard bytes. Both default and `HTP_FAST_CONCAT=1` outputs are checked.
The marker is emitted only on actual fast-path execution, once per executor.

Both executor translation units are compiled with UndefinedBehaviorSanitizer.
GGML base is linked as a normal dependency because its existing graph-size
helper deliberately offsets a null pointer. The hybrid executor's five
otherwise duplicated support symbols receive test-only names during compilation;
concat arithmetic and scheduling code remain the production implementation.

The capability fixture also tests the v68 flash-attention head dimension bound:
512 is supported;576 and768 retain CPU support.

All optimization flags remain off by default. These host checks establish copy
and routing correctness; phone inference quality and speed are separate tests.
