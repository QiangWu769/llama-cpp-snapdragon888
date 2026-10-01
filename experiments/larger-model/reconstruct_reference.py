#!/usr/bin/env python3
"""Correctness helper: HTP Q4_0+F16 weights -> ordinary CPU weights.

The input must have been made by convert_hf_to_gguf_htp.py followed by
llama-quantize Q4_0+F16 with REPACK_FOR_HVX set. GGUF does not encode that
private packing distinction, so explicit --confirm-htp-hvx-packed is required.
This reverses storage permutations; it never requantizes the source weights.
Q4_0 weights become their HMX FP16 dequantization products, already rounded
once to binary16. Existing packed F16 matrices become ordinary F16. Ordinary
CPU tensors (including the tied Q6_K embedding/output) remain byte-identical.
CPU activations/accumulation still differ from HMX FP16, so inference results
need a numerical comparison, not necessarily bitwise identical logits.

Usage (gguf is imported from the requested checkout):
  python reconstruct_reference.py --repo /path/to/checkout \
    --self-test
  python reconstruct_reference.py --repo /path/to/checkout \
    --confirm-htp-hvx-packed packed.gguf cpu-reference.gguf
The output and .manifest.json sidecar must not exist. No external model code
is imported or executed; only the checkout's GGUF data reader/writer is used.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


PACKED_MATRIX = re.compile(
    r"blk\.\d+\.(?:attn_q|attn_k|attn_v|attn_output|ffn_up|ffn_down|ffn_gate)\.weight\Z"
)


def unpack_f16_tiles(packed: np.ndarray, n: int, k: int) -> np.ndarray:
    """Reverse precisely Model.modify_tensors' two tensor-axis permutations."""
    if n <= 0 or k <= 0 or n % 32 or k % 32:
        raise ValueError(f"Unsupported packed matrix shape {(n, k)}")
    if packed.size != n * k:
        raise ValueError("Packed FP16 element count does not match matrix shape")
    nc, kc = n // 32, k // 32
    tiles = packed.reshape(nc, kc, 16, 32, 2).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(
        tiles.reshape(nc, kc, 32, 32).transpose(0, 2, 1, 3).reshape(n, k)
    )


def inverse_q4_hvx_superpack(data: np.ndarray) -> np.ndarray:
    """Return ordinary 18-byte GGML Q4_0 blocks from 144-byte superblocks."""
    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    if raw.size % 144:
        raise ValueError("HVX Q4_0 byte count is not a multiple of 144")
    superblocks = raw.reshape(-1, 144)
    output = np.empty((superblocks.shape[0], 8, 18), dtype=np.uint8)
    output[:, :, :2] = superblocks[:, :16].reshape(-1, 8, 2)
    interleaved = superblocks[:, 16:].reshape(-1, 64, 2)
    codes = np.empty((superblocks.shape[0], 256), dtype=np.uint8)
    codes[:, :64] = interleaved[:, :, 0] & 15
    codes[:, 64:128] = interleaved[:, :, 1] & 15
    codes[:, 128:192] = interleaved[:, :, 0] >> 4
    codes[:, 192:] = interleaved[:, :, 1] >> 4
    codes = codes.reshape(-1, 8, 32)
    output[:, :, 2:] = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return output.reshape(-1, 18)


def dequantize_q4_hmx_fp16(data: np.ndarray) -> np.ndarray:
    blocks = inverse_q4_hvx_superpack(data)
    # A binary16 scale times an integer in [-8,7] fits exactly in binary32;
    # casting the exact product implements the HMX path's ties-even rounding.
    scales = np.ascontiguousarray(blocks[:, :2]).view("<f2").astype(np.float32)
    quants = blocks[:, 2:]
    codes = np.concatenate((quants & 15, quants >> 4), axis=1).astype(np.int16)
    products = scales * (codes - 8).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        result = products.astype(np.float16)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite dequantized matrix; finite CPU reference required")
    return result.reshape(-1)


def decode_metadata(field, gguf):
    vtype = field.types[0]
    if vtype == gguf.GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8")
    if vtype == gguf.GGUFValueType.ARRAY:
        # ReaderField.types omits the element type for an empty array.
        # parts[3] is the on-disk array subtype after name-length/name/KV-type.
        subtype = gguf.GGUFValueType(int(field.parts[3][0]))
        if subtype == gguf.GGUFValueType.ARRAY:
            raise ValueError("Nested metadata arrays are unsupported")
        if subtype == gguf.GGUFValueType.STRING:
            values = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        else:
            values = [value for i in field.data for value in field.parts[i].tolist()]
        return TypedArray(values, subtype)
    return field.parts[field.data[0]][0].item()


class TypedArray(list):
    """Carry the original GGUF array subtype, including empty arrays."""
    def __init__(self, values, subtype):
        super().__init__(values)
        self.subtype = subtype


def writer_class(gguf):
    class PreservingWriter(gguf.GGUFWriter):
        def _pack_val(self, val, vtype, add_vtype):
            if vtype != gguf.GGUFValueType.ARRAY or not isinstance(val, TypedArray):
                return super()._pack_val(val, vtype, add_vtype)
            prefix = self._pack("I", vtype) if add_vtype else b""
            prefix += self._pack("I", val.subtype) + self._pack("Q", len(val))
            return prefix + b"".join(
                super(PreservingWriter, self)._pack_val(item, val.subtype, False)
                for item in val
            )
    return PreservingWriter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def reconstruct(input_path: Path, output_path: Path, gguf) -> dict:
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    for target in (output_path, manifest_path, partial):
        if target.exists():
            raise FileExistsError(f"Refusing to replace {target}")
    reader = gguf.GGUFReader(input_path)
    if reader.byte_order != "I" or sys.byteorder != "little":
        raise ValueError("This helper supports little-endian input/host only")
    if any(name.startswith("split.") for name in reader.fields):
        raise ValueError("Split GGUF input is unsupported")
    architecture = decode_metadata(reader.get_field("general.architecture"), gguf)
    if architecture != "qwen2":
        raise ValueError("This validated helper is scoped to Qwen2")
    writer = writer_class(gguf)(partial, architecture)
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        writer.add_key_value(name, decode_metadata(field, gguf), field.types[0])
    writer.data_alignment = int(reader.alignment)
    records = []
    for tensor in reader.tensors:
        transform = bool(PACKED_MATRIX.fullmatch(tensor.name))
        if transform:
            if len(tensor.shape) != 2:
                raise ValueError(f"Not a matrix: {tensor.name}")
            k, n = map(int, tensor.shape)
            if k % 32 or n % 32:
                raise ValueError(f"Unaligned tensor: {tensor.name}")
            if tensor.tensor_type not in (
                gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.F16
            ):
                raise ValueError(f"Unexpected packed type {tensor.tensor_type}: {tensor.name}")
            writer.add_tensor_info(
                tensor.name, (n, k), np.dtype(np.float16), tensor.n_elements * 2
            )
        else:
            # Any other quantized matrix could itself be superpacked. Fail
            # instead of silently forwarding data whose packing is ambiguous.
            if tensor.tensor_type not in (
                gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.Q6_K,
            ):
                raise ValueError(f"Unexpected ordinary type {tensor.tensor_type}: {tensor.name}")
            if tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K and tensor.name not in (
                "token_embd.weight", "output.weight"
            ):
                raise ValueError(f"Q6_K outside ordinary embedding/output: {tensor.name}")
            writer.add_tensor_info(
                tensor.name, tensor.data.shape, tensor.data.dtype, tensor.n_bytes,
                raw_dtype=tensor.tensor_type,
            )
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            transform = bool(PACKED_MATRIX.fullmatch(tensor.name))
            if transform:
                k, n = map(int, tensor.shape)
                if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0:
                    packed = dequantize_q4_hmx_fp16(tensor.data)
                    operation = "inverse HVX superpack; Q4_0 dequantize to HMX FP16; inverse tiles"
                else:
                    packed = tensor.data
                    operation = "inverse FP16 tiles"
                output = unpack_f16_tiles(packed, n, k)
            else:
                output = tensor.data
                operation = "byte-preserving copy"
            writer.write_tensor_data(output)
            records.append({"name": tensor.name, "input_type": tensor.tensor_type.name,
                            "output_type": "F16" if transform else tensor.tensor_type.name,
                            "shape_ggml": tensor.shape.tolist(), "operation": operation,
                            "output_bytes": output.nbytes,
                            "output_sha256": hashlib.sha256(output.tobytes()).hexdigest()})
            print(f"{tensor.name}: {operation}", flush=True)
        writer.close()
        partial.replace(output_path)
    except BaseException:
        writer.close()
        raise
    result = {
        "input": str(input_path), "input_sha256": sha256_file(input_path),
        "output": str(output_path), "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "metadata_policy": "Preserve all source KV names, values, scalar types and array subtypes; general.file_type is descriptive source metadata, tensor directory has authoritative types.",
        "reference_contract": "Same HMX FP16 dequantized weights in ordinary CPU layout; CPU activations/accumulation still differ.",
        "tensors": records,
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def self_test(repo: Path, gguf):
    rng = np.random.default_rng(124568)
    for n, k in ((32, 32), (64, 96), (96, 64), (32, 8960)):
        logical = rng.uniform(-1, 1, (n, k)).astype(np.float16)
        # Literal converter operations, independently of inverse function.
        x = logical.reshape(n // 32, 32, k // 32, 32).transpose(0, 2, 1, 3).copy()
        packed = x.reshape(n // 32, k // 32, 32, 16, 2).transpose(0, 1, 3, 2, 4).copy()
        restored = unpack_f16_tiles(packed, n, k)
        np.testing.assert_array_equal(restored.view(np.uint16), logical.view(np.uint16))

    # Compile the actual production repacker source, not a second Python copy.
    source = (repo / "llama.cpp-npu/ggml/src/ggml-quants.c").read_text()
    start = source.index("void repack_q4_0_super_block_hvx(const block_q4_0 * src, void * dst, size_t size) {")
    end = source.index("\nvoid repack_q8_0_super_block_hvx", start)
    prelude = "#include <stdint.h>\n#include <stddef.h>\n#include <string.h>\n#include <assert.h>\n#define QK4_0 32\ntypedef uint16_t ggml_fp16_t;\ntypedef struct { ggml_fp16_t d; uint8_t qs[16]; } block_q4_0;\n"
    with tempfile.TemporaryDirectory(prefix="htp-reference-test-") as temporary:
        directory = Path(temporary)
        c_path = directory / "repacker.c"
        so_path = directory / "repacker.so"
        c_path.write_text(prelude + source[start:end])
        subprocess.run(["cc", "-std=c11", "-shared", "-fPIC", "-O2", str(c_path), "-o", str(so_path)], check=True)
        library = ctypes.CDLL(str(so_path))
        repack = library.repack_q4_0_super_block_hvx
        repack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        repack.restype = None
        standard = rng.integers(0, 256, size=(8 * 128, 18), dtype=np.uint8)
        # Include signed zero, subnormal, normal negative/positive scales.
        scales = rng.uniform(-0.25, 0.25, len(standard)).astype(np.float16)
        scales[:6] = [0.0, -0.0, np.nextafter(np.float16(0), np.float16(1)),
                      -np.nextafter(np.float16(0), np.float16(1)), 0.0173, -0.0197]
        standard[:, :2] = scales.view(np.uint8).reshape(-1, 2)
        packed = np.empty_like(standard)
        repack(standard.ctypes.data, packed.ctypes.data, standard.nbytes)
        np.testing.assert_array_equal(inverse_q4_hvx_superpack(packed), standard)
        expected = gguf.dequantize(standard, gguf.GGMLQuantizationType.Q4_0).astype(np.float16).reshape(-1)
        actual = dequantize_q4_hmx_fp16(packed)
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

        # End-to-end GGUF directory, source metadata subtype and ordinary Q6_K
        # byte-copy checks use a tiny synthetic Qwen2 container.
        input_path, output_path = directory / "packed.gguf", directory / "cpu.gguf"
        writer = writer_class(gguf)(input_path, "qwen2")
        writer.add_uint32("general.file_type", 37)
        writer.add_key_value("fixture.uint64_array", TypedArray([2**42, 2**42+1], gguf.GGUFValueType.UINT64), gguf.GGUFValueType.ARRAY)
        writer.add_key_value("fixture.empty_string_array", TypedArray([], gguf.GGUFValueType.STRING), gguf.GGUFValueType.ARRAY)
        writer.add_array("tokenizer.ggml.tokens", ["a", "b"])
        writer.add_tensor("blk.0.attn_q.weight", packed.reshape(32, 32*18), raw_dtype=gguf.GGMLQuantizationType.Q4_0)
        logical = rng.uniform(-1, 1, (32, 64)).astype(np.float16)
        x = logical.reshape(1, 32, 2, 32).transpose(0, 2, 1, 3).copy()
        tiles = x.reshape(1, 2, 32, 16, 2).transpose(0, 1, 3, 2, 4).copy().reshape(32, 64)
        writer.add_tensor("blk.0.ffn_down.weight", tiles)
        q6_bytes = rng.integers(0, 256, (2, 210), dtype=np.uint8)
        writer.add_tensor("token_embd.weight", q6_bytes, raw_dtype=gguf.GGMLQuantizationType.Q6_K)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reconstruct(input_path, output_path, gguf)
        output_reader = gguf.GGUFReader(output_path)
        tensors = {tensor.name: tensor for tensor in output_reader.tensors}
        np.testing.assert_array_equal(tensors["blk.0.attn_q.weight"].data.view(np.uint16), unpack_f16_tiles(actual, 32, 1024).view(np.uint16))
        np.testing.assert_array_equal(tensors["blk.0.ffn_down.weight"].data.view(np.uint16), logical.view(np.uint16))
        np.testing.assert_array_equal(tensors["token_embd.weight"].data, q6_bytes)
        original = gguf.GGUFReader(input_path)
        for name, field in original.fields.items():
            if name.startswith("GGUF."):
                continue
            copied = output_reader.get_field(name)
            assert copied.types == field.types, (name, copied.types, field.types)
            assert decode_metadata(copied, gguf) == decode_metadata(field, gguf), name
    print("PASS: 4 FP16 tile shapes; 1024 exact C-repacked Q4_0 blocks; FP16 dequant bits; streamed GGUF metadata/types; Q6_K byte-copy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--confirm-htp-hvx-packed", action="store_true")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "llama.cpp-npu/gguf-py"))
    import gguf
    if args.self_test:
        self_test(args.repo, gguf)
        return
    if not args.confirm_htp_hvx_packed or args.input is None or args.output is None:
        parser.error("Conversion requires --confirm-htp-hvx-packed and input/output paths")
    result = reconstruct(args.input, args.output, gguf)
    print(json.dumps({key: value for key, value in result.items() if key != "tensors"}, indent=2))


if __name__ == "__main__":
    main()
