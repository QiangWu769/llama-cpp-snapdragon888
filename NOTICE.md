# Upstream provenance and notices

This repository combines pinned source snapshots of two upstream projects and applies a Snapdragon 888 / Hexagon v68 research port. See `UPSTREAM.json` for exact URLs and revisions. The first commit imports those sources; subsequent commits contain the adaptation and its documentation.

- `llama.cpp-npu/` derives from haozixu/llama.cpp-npu and llama.cpp. Its existing MIT license and third-party notices are retained in that subtree.
- `htp-ops-lib/` derives from haozixu/htp-ops-lib. The pinned upstream snapshot has no top-level LICENSE file. Existing file-level notices are retained, including Qualcomm notices. This repository does not assign a replacement blanket license to that subtree or claim ownership of upstream code.
- Hexagon SDK, QNN binaries, firmware libraries, model weights, credentials and compiled device binaries are not distributed here. Build against your separately obtained SDK and device runtime.
- The full upstream snapshots retain their existing development assets and tests, including 16 GGUF vocabulary fixtures with zero weight tensors, the Android Gradle wrapper, and presentation/web assets. Existing upstream Qualcomm-attributed source excerpts retain their notices; this port adds no SDK installation or SDK runtime package.

Upstream research: *Scaling LLM Test-Time Compute with Mobile NPU on Smartphones*, Zixu Hao et al., arXiv:2509.23324. This port is an independent adaptation of that implementation, not an upstream endorsement. Keep the upstream attribution when referring to the underlying methods.
