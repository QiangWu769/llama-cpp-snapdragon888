# Larger-model research functions

The [report](../../docs/LARGER-MODEL.md) describes the tested Qwen2.5-1.5B
mixed-precision model, wider-K F16 planner, host DMA mapping failures and
phase-aligned activity measurements.

`reconstruct_reference.py` reconstructs ordinary CPU-layout layer weights
from the actual HTP/HVX packed Q4_0+F16 model. It preserves Q6_K embedding
bytes and rounds dequantized layer products to the FP16 values supplied to
HMX. It does not independently requantize weights. `reconstruct()` and its
packing functions are importable; `--self-test` checks the inverse against
the production C repacker and GGUF directory/metadata handling.

`export_evidence.export(root, destination)` requires all eight completed
activity captures, checks outcomes, analyzes corrected intervals and writes
measured values without SDK binaries or database contents. It retains CPU
output differences and coverage. Optional numerical trial directories retain
the CPU/HMX probability and greedy diagnostics, the CPU-attention isolation,
failed setup/mapping outcomes and final device checks. It also checks that the
captured Android collector source hash matches the public library caller.

`teacher-forced-input.txt` is the fixed English numerical fixture. The
perplexity caller saves scaled uint16 log probabilities, not raw FP32 logits.
Run diagnostic callers separately from activity benchmarks and preserve failed
attempts. See the report for measured outcomes.

Inference uses the existing [C++ library caller](../load-calibration/inference/).
Collection uses the existing importable phone-local
[`collect()` function](../load-calibration/capture_workload.py), which calls
`qcphoneperf.Profiler`. The development host stages and retrieves results; the
collector and inference run on the phone without a runtime computer service,
collector CLI or APK.
