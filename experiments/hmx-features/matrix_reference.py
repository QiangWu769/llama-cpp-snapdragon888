#!/usr/bin/env python3
"""Independent, exact-dyadic matrix fixtures; no DSP/SDK/phone dependencies."""
import argparse
import hashlib
import json
import struct


def half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def spatial_bits(value):
    return ((value >> 1) << 7) | ((value & 1) << 1)


def pack_activation(rows):
    packed = [0] * 1024
    for r in range(32):
        for k in range(32):
            packed[(r // 2) * 64 + 2 * k + r % 2] = half_bits(rows[r][k])
    return packed


def pack_weights(rows):
    assert len(rows) % 2 == 0
    packed = [0] * (32 * len(rows))
    for k, row in enumerate(rows):
        for n, value in enumerate(row):
            packed[(k // 2) * 64 + 2 * n + k % 2] = half_bits(value)
    return packed


def activation(tile):
    return [[(((r * 3 + k * 5 + tile * 7) % 9) - 4) / 4
             for k in range(32)] for r in range(32)]


def weights(tile):
    return [[(((k * 3 + n * 5 + tile * 11) % 7) - 3) / 4
             for n in range(32)] for k in range(32)]


def dot(a, b):
    return [[sum(x * b[k][n] for k, x in enumerate(row))
             for n in range(32)] for row in a]


def flatten_half(matrix):
    return [half_bits(x) for row in matrix for x in row]


def fixture(name, kind, tiles=1, first=0, last=31,
            mask=28, y_shift=0, displacement=2048, bit0=0):
    acts = [activation(t) for t in range(tiles)]
    groups = 2 if kind == "weight_deep" else 1
    if kind == "single":
        assert mask == 28  # The independent reference uses X=4, Y=8.
        assert tiles == 2
        # Logical tile order is independent of physical placement/dY sign.
        effective = []
        for r in range(32):
            y, x = divmod(r, 4)
            shifted = y + y_shift
            effective.append(acts[shifted // 8][(shifted % 8) * 4 + x][first:last + 1])
        selected_b = weights(0)[first:last + 1]
        ref = dot(effective, selected_b)
        packed_b = pack_weights(selected_b)
        upper = displacement & 0xffffffff
        offset = y_shift << 2
    elif kind == "weight_deep":
        ref_parts = [dot(acts[0], weights(t)) for t in range(2)]
        ref = [ref_parts[0][r] + ref_parts[1][r] for r in range(32)]
        packed_b = pack_weights(weights(0)) + pack_weights(weights(1))
        upper, offset = 0, 0
    else:
        begin, end = first, (tiles - 1) * 32 + last + 1
        a_long = [sum((tile[r] for tile in acts), []) for r in range(32)]
        b_long = sum((weights(t) for t in range(tiles)), [])
        ref = dot([row[begin:end] for row in a_long], b_long[begin:end])
        packed_b = pack_weights(b_long[begin:end])
        upper, offset = (tiles - 1) << 11, 0
    if kind == "single" and displacement < 0:
        tile_offsets = [-displacement, 0]
    elif kind == "single":
        tile_offsets = [0, displacement]
    else:
        tile_offsets = [2048 * t for t in range(tiles)]
    rs_low = (first << 2) | spatial_bits(offset)
    rt = upper | spatial_bits(mask) | (last << 2) | bit0
    return {
        "name": name, "kind": kind, "spatial_mask": mask,
        "activation_tiles": tiles, "filter_groups": groups,
        "channel_first": first, "channel_last": last,
        "activation_base_offset": tile_offsets[0],
        "activation_rs_low": rs_low, "activation_rt": rt,
        "weight_rt": len(packed_b) * 2 - 1,
        "logical_activation_tile_byte_offsets": tile_offsets,
        "activation_tile_half_bits": [pack_activation(a) for a in acts],
        "weight_half_bits": packed_b,
        "expected_row_major_half_bits": flatten_half(ref),
        "expected_row_major_float": sum(ref, []),
        "output_columns": 32 * groups,
        "y_shift": y_shift, "dY": displacement if kind == "single" else None,
    }


def all_fixtures():
    yield fixture("baseline_mask28", "regular")
    yield fixture("baseline_mask31", "regular", mask=31)
    yield fixture("baseline_legacy_bit0", "regular", mask=31, bit0=1)
    for first, last in [(0, 7), (8, 15), (16, 31), (24, 31)]:
        yield fixture(f"partial_{first}_{last}", "regular", first=first, last=last)
    for tiles in (1, 2, 3, 32):
        yield fixture(f"deep_{tiles}", "deep", tiles=tiles)
    yield fixture("deep_partial_3_24_15", "deep", tiles=3, first=24, last=15)
    yield fixture("filters64", "weight_deep")
    for y_shift, displacement in [(0, 2048), (1, 2048), (5, 2048),
                                  (7, 2048), (5, 4096), (5, -2048)]:
        yield fixture(f"single_y{y_shift}_d{displacement}", "single", tiles=2,
                      y_shift=y_shift, displacement=displacement)
    yield fixture("single_partial_y5_8_15", "single", tiles=2,
                  y_shift=5, first=8, last=15)


def self_check(cases):
    assert spatial_bits(28) | (31 << 2) == 0x77c
    assert spatial_bits(31) | (31 << 2) == 0x7fe
    assert len(cases) == 20
    by_name = {c["name"]: c for c in cases}
    assert by_name["deep_3"]["activation_rt"] == 0x177c
    assert by_name["deep_partial_3_24_15"]["weight_rt"] == 3583
    assert by_name["single_y5_d2048"]["activation_rs_low"] == 0x500
    assert by_name["single_y5_d-2048"]["activation_rt"] == 0xffffff7c
    assert by_name["filters64"]["weight_rt"] == 4095
    assert by_name["baseline_mask28"]["expected_row_major_half_bits"] == \
        by_name["deep_1"]["expected_row_major_half_bits"]
    assert by_name["single_y0_d2048"]["expected_row_major_half_bits"] == \
        by_name["baseline_mask28"]["expected_row_major_half_bits"]
    for c in cases:
        assert len(c["weight_half_bits"]) * 2 == c["weight_rt"] + 1
        assert len(c["expected_row_major_half_bits"]) == 32 * c["output_columns"]
        # These fixtures deliberately avoid uncertain accumulator/rounding edges.
        for value, bits in zip(c["expected_row_major_float"], c["expected_row_major_half_bits"]):
            assert value == struct.unpack("<e", struct.pack("<H", bits))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", help="Emit one named fixture; 'all' emits all.")
    parser.add_argument("--out", help="Output path for JSON fixture data.")
    args = parser.parse_args()
    cases = list(all_fixtures())
    self_check(cases)
    if args.emit:
        if not args.out:
            parser.error("--emit requires --out")
        data = cases if args.emit == "all" else next((c for c in cases if c["name"] == args.emit), None)
        if data is None:
            parser.error("Unknown case")
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
    else:
        for c in cases:
            digest = hashlib.sha256(struct.pack("<" + "H" * len(c["expected_row_major_half_bits"]),
                                               *c["expected_row_major_half_bits"])).hexdigest()[:12]
            print(f'{c["name"]:29} act_rs_low=0x{c["activation_rs_low"]:03x} '
                  f'act_rt=0x{c["activation_rt"]:08x} weight_rt={c["weight_rt"]:5} ref={digest}')
        print(f"PASS: {len(cases)} independent fixtures; all reference outputs exactly representable in FP16.")


if __name__ == "__main__":
    main()
