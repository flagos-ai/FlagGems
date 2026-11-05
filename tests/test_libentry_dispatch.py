# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep cache-key classification and fallback binding identical across dispatch."""

import importlib

import pytest
import torch
import triton
import triton.language as tl

from flag_gems.utils.libentry import LibEntry

runtime = importlib.import_module("flag_gems.utils.libentry")


@triton.jit(do_not_specialize=["n", "scale"])
def interleaved(X, n, SHIFT: tl.constexpr, Y, scale, optional, BLOCK: tl.constexpr):
    pass


@pytest.mark.parametrize("minor", [2, 3, 6, 7])
@pytest.mark.parametrize("flagtune", [False, True])
@pytest.mark.parametrize("style", ["positional", "mixed", "keywords"])
def test_dispatch_classification_cold_warm_and_new_key(
    monkeypatch, minor, flagtune, style
):
    entry = LibEntry(interleaved)
    monkeypatch.setattr(runtime, "major_version", 3)
    monkeypatch.setattr(runtime, "minor_version", minor)
    monkeypatch.setattr(runtime.torch_device_fn, "current_device", lambda: "cpu")
    entry._has_flagtune_tuner = flagtune
    applied, dtype_inputs, keys, cold, launches = [], [], [], [], []
    monkeypatch.setattr(entry, "_apply_flagtune", lambda: applied.append(True))

    def infer(values):
        dtype_inputs.append(tuple(values))
        return (torch.float32, torch.float32)

    monkeypatch.setattr(runtime, "_infer_tensor_dtypes", infer)
    original_key = entry.key

    def record_key(spec, dns, const):
        keys.append((list(spec), list(dns), list(const)))
        return original_key(spec, dns, const)

    monkeypatch.setattr(entry, "key", record_key)

    class Compiled:
        def __getitem__(self, grid):
            def launch(*args):
                launches.append((grid, args))

            return launch

    compiled = Compiled()

    def first(*args, **kwargs):
        cold.append((args, kwargs))
        return compiled

    monkeypatch.setattr(entry, "_first_run", first)
    seen_grids = []

    def grid(meta):
        seen_grids.append(meta)
        return (triton.cdiv(meta["n"], meta["BLOCK"]),)

    for index, (n, shift) in enumerate([(65, 7), (257, 7), (130, 9)]):
        x, y = torch.ones(4), torch.zeros(4)
        args = (x, n, shift, y, float(index + 2), None, 128)
        kwargs = dict(zip(entry.arg_names, args))
        if style == "positional":
            entry.run(*args, grid=grid)
        elif style == "mixed":
            entry.run(*args[:2], **dict(list(kwargs.items())[2:]), grid=grid)
        else:
            entry.run(**kwargs, grid=grid)
        spec, dns, const = keys[-1]
        assert spec[0] is x and spec[1] is y and spec[2] is None
        assert dns == [n, float(index + 2)]
        assert const[:2] == [shift, 128]
        if flagtune:
            assert const[2:] == [("flagtune_dtypes", "torch.float32", "torch.float32")]
            bound = dtype_inputs[-1]
            # On unsupported versions the mapping excludes positional constexprs,
            # and keyword constexprs too; retain that historical behavior.
            expected = args if 3 <= minor <= 6 else (x, n, y, float(index + 2), None)
            assert len(bound) == len(expected)
            for actual, wanted in zip(bound, expected):
                assert actual is wanted or (
                    not isinstance(actual, torch.Tensor) and actual == wanted
                )
        else:
            assert const == [shift, 128]
        if index == 1:
            launch_grid, bound = launches[-1]
            expected = args if 3 <= minor <= 6 else (x, n, y, float(index + 2), None)
            assert launch_grid == (3, 1, 1)
            assert len(bound) == len(expected)
            for actual, wanted in zip(bound, expected):
                assert actual is wanted or (
                    not isinstance(actual, torch.Tensor) and actual == wanted
                )
            assert seen_grids[-1]["X"] is x
    assert len(cold) == 2  # runtime values reuse a key; constexpr change does not
    assert len(launches) == 1
    assert len(applied) == (3 if flagtune else 0)
    assert len(dtype_inputs) == (3 if flagtune else 0)
