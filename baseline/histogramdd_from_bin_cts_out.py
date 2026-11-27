# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA native baseline for `histogramdd_from_bin_cts_out`.

Mirrors benchmark/test_histogramdd_from_bin_cts.py
(HistogramddBenchmark with the local `baseline` wrapper defined inside
test_histogramdd_from_bin_cts_out):
    op_name  = "histogramdd_from_bin_cts_out"
    torch_op = a CPU wrapper around
               torch.ops.aten._histogramdd_from_bin_cts.out
    dtypes   = [torch.float32]

CAUTION: this operator has no native CUDA kernel. The benchmark's baseline
(`_make_cpu_baseline`) moves the input to the host, runs the CPU reference and
moves the result back, so `latency_base` is a CPU measurement plus two
transfers and NCU will find no device kernel for it. That CPU-vs-GPU comparison
is deliberate on the benchmark's part -- PyTorch ships this operator for CPU
only. The same wrapper is reproduced here. See Q&A.md.

HistogramddBenchmark pins self.shapes to POINT_SHAPES in init_user_config, so
core_shapes.yaml is overridden. bins is [20] * ndim and range is
[-30.0, 30.0] * ndim, where ndim is the last dim; the point cloud is
randn(shape) * 10.

The .out wrapper allocates the host output as empty(tuple(bins)) in the input
dtype, calls the .out overload on the CPU and copies the result back to the
input's device.
"""

import torch

OP_NAME = "histogramdd_from_bin_cts_out"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.ops.aten._histogramdd_from_bin_cts.out is unavailable on this build"
)

# test_histogramdd_from_bin_cts.py::POINT_SHAPES -- the last dim is the
# number of histogram dimensions, so it stays small while the point count grows.
_SHAPES = [
    (1024, 2),
    (10000, 2),
    (100000, 2),
    (1000, 3),
    (50000, 3),
    (100000, 3),
    (10000, 1),
    (1000000, 1),
]

_BINS_PER_DIM = 20
_RANGE_LO = -30.0
_RANGE_HI = 30.0


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _cpu_baseline_out(inp, *, bins, range=None, weight=None, density=False, **kw):
    """benchmark::test_histogramdd_from_bin_cts_out::baseline."""
    inp_cpu = inp.detach().to("cpu")
    weight_cpu = weight.detach().to("cpu") if weight is not None else None
    out_shape = tuple(bins)
    out = torch.empty(out_shape, dtype=inp.dtype, device="cpu")
    torch.ops.aten._histogramdd_from_bin_cts.out(
        inp_cpu, bins, range=range, weight=weight_cpu, density=density, out=out
    )
    return out.to(inp.device)


def native():
    if _resolve("torch.ops.aten._histogramdd_from_bin_cts.out") is None:
        return None
    return _cpu_baseline_out


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device) * 10
    ndim = shape[-1]
    bins = [_BINS_PER_DIM] * ndim
    return (inp,), {"bins": bins, "range": [_RANGE_LO, _RANGE_HI] * ndim}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    ndim = shape[-1]
    dt = str(dtype)
    bins = [_BINS_PER_DIM] * ndim
    return {
        "inputs": {
            "args": [{"shape": shape, "dtype": dt}],
            "kwargs": {
                "bins": bins,
                "range": [_RANGE_LO, _RANGE_HI] * ndim,
            },
        },
        "outputs": {"out": {"shape": bins, "dtype": dt}},
        "dims": {"points": shape[0], "ndim": ndim, "bins": bins},
    }
