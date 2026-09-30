#!/usr/bin/env python3
"""Exact dyadic fixtures for V81 epilogue semantics; not a v68 emulator.

Print candidate bias words and finite expected values. The input is an exact
accumulator value. This intentionally does not model undocumented accumulator
rounding, special-value behavior, or any legacy bias encoding.
"""
from fractions import Fraction as F
import json
import struct


def hf_bits(value):
    return int.from_bytes(struct.pack('<e', float(value)), 'little')


def from_hf(bits):
    return F(struct.unpack('<e', int(bits).to_bytes(2, 'little'))[0])


def rounded_hf(value):
    return from_hf(hf_bits(value))


def pack_bias(input_bias=0, shape=0, scale=1, output_bias=0,
              input_extra=0, scale_extra=0, output_extra=0):
    assert 0 <= shape <= 7
    assert 0 <= input_extra < 32
    assert 0 <= scale_extra < 16 and 0 <= output_extra < 16
    value = ((hf_bits(input_bias) << 5 | input_extra) << 43)
    value |= shape << 40
    value |= output_extra << 36 | scale_extra << 32
    value |= hf_bits(output_bias) << 16 | hf_bits(scale)
    return {'low': f'0x{value & 0xffffffff:08x}',
            'high': f'0x{value >> 32:08x}'}


def shape_fn(x, shape):
    functions = [lambda v: v, lambda v: min(v, 0),
                 lambda v: max(v, 0), abs]
    y = functions[shape & 3](x)
    return -y if shape & 4 else y


def convert(x, input_bias=0, shape=0, scale=1, output_bias=0,
            previous=None, destination=None, operation=None):
    scale, output_bias = F(scale), F(output_bias)
    if destination:
        assert previous is not None and operation in ('min', 'max')
        limit = min if operation == 'min' else max
        if destination == 'scale':
            scale = limit(F(previous), scale)
        elif destination == 'output_bias':
            output_bias = limit(F(previous), output_bias)
        else:
            raise ValueError(destination)
    return scale * shape_fn(F(x) + F(input_bias), shape) + output_bias


def fixtures():
    rows = []
    x_values = [F(-2), F(-1, 2), F(-1, 4), F(0),
                F(1, 4), F(1, 2), F(2), F(8)]
    for shape in range(8):
        params = dict(input_bias=F(1, 4), shape=shape,
                      scale=F(3, 2), output_bias=F(-1, 2))
        rows.append(dict(kind='shape', shape=shape, bias=pack_bias(**params),
                         x=[float(x) for x in x_values],
                         expected_half=[f'0x{hf_bits(convert(x, **params)):04x}'
                                        for x in x_values]))
    for dest in ('scale', 'output_bias'):
        for op in ('min', 'max'):
            params = dict(input_bias=F(1, 4), scale=F(3, 2), output_bias=F(3, 4))
            rows.append(dict(kind='feedback', destination=dest, operation=op,
                             bias=pack_bias(**params),
                             expected_half=[f'0x{hf_bits(convert(x, previous=x, destination=dest, operation=op, **params)):04x}'
                                            for x in x_values]))
    x = F(1) + F(1, 4096)
    assert rounded_hf(x) == 1
    assert hf_bits(x * rounded_hf(x)) == 0x3c00
    assert hf_bits(x * x) == 0x3c01
    rows.append(dict(kind='feedback_extra_precision', accumulator=float(x),
                     seed_control_rnd0='0x001', seed_control_rnd1='0x101',
                     second_control='0x018',
                     second_bias=pack_bias(scale=0),
                     expected_rnd0='0x3c00', expected_rnd1='0x3c01'))
    # Exact cancellation amplifies the extended bias fields into FP16 outputs.
    rows.extend([
        dict(kind='input_bias_extra', accumulator=-1,
             bias=pack_bias(input_bias=1, input_extra=1, scale=1024),
             expected_half=f'0x{hf_bits(F(1, 32)):04x}'),
        dict(kind='scale_extra', accumulator=1024,
             bias=pack_bias(scale=1, scale_extra=1, output_bias=-1024),
             expected_half=f'0x{hf_bits(F(1, 16)):04x}'),
        dict(kind='output_bias_extra', accumulator=-1,
             bias=pack_bias(scale=1, output_bias=1, output_extra=1),
             expected_half=f'0x{hf_bits(F(1, 16384)):04x}')])
    return rows


if __name__ == '__main__':
    assert pack_bias() == {'low': '0x00003c00', 'high': '0x00000000'}
    print(json.dumps(fixtures(), indent=2))
