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

import pytest
import torch

import flag_gems

from . import base


def _gems_qr_out(A, out=None, mode="reduced"):
    # adapt torch.linalg.qr's `out=(Q, R)` convention to the gems op signature
    Q, R = out
    return flag_gems.linalg_qr_out(A, mode, Q=Q, R=R)


# All three torch.linalg.qr modes are benchmarked (the accuracy tests cover
# the same set): reduced (default), complete, r (R only).  For wide inputs
# (m < n) complete coincides with reduced (k = m), so every mode is valid
# for every shape.
# Shapes come from the linalg_qr / linalg_qr_out entries in core_shapes.yaml.
QR_MODES = ["reduced", "complete", "r"]

# fp64 is only benchmarked on backends that support it.
QR_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    QR_DTYPES.append(torch.float64)


def _qr_out_shapes(shape, mode):
    """Output (Q, R) shapes of torch.linalg.qr's out= variant per mode."""
    *batch, m, n = shape
    k = min(m, n)
    if mode == "complete":
        return (*batch, m, m), (*batch, m, n)
    if mode == "r":
        return (0,), (*batch, k, n)
    return (*batch, m, k), (*batch, k, n)


def _make_qr_input_fn(mode):
    def input_fn(shape, dtype, device):
        shape = tuple(shape)
        yield torch.randn(shape, dtype=dtype, device=device), {"mode": mode}

    return input_fn


def _make_qr_out_input_fn(mode):
    def input_fn(shape, dtype, device):
        shape = tuple(shape)
        A = torch.randn(shape, dtype=dtype, device=device)
        Qshape, Rshape = _qr_out_shapes(shape, mode)
        Q = torch.empty(Qshape, dtype=dtype, device=device)
        R = torch.empty(Rshape, dtype=dtype, device=device)
        yield A, {"out": (Q, R), "mode": mode}

    return input_fn


class QRGenericBenchmark(base.GenericBenchmark):
    """GenericBenchmark for linalg_qr with shapes from core_shapes.yaml only.

    The base class merges generic 1D/2D/3D extra shapes (e.g. (2**28,)) that
    are not valid QR inputs (A must have at least 2 dimensions), so the
    generic extras are disabled here.
    """

    def set_more_shapes(self):
        return []


@pytest.mark.linalg_qr
@pytest.mark.parametrize("mode", QR_MODES)
def test_linalg_qr(mode):
    bench = QRGenericBenchmark(
        op_name="linalg_qr",
        input_fn=_make_qr_input_fn(mode),
        torch_op=torch.ops.aten.linalg_qr,
        gems_op=flag_gems.linalg_qr,
        dtypes=QR_DTYPES,
    )
    bench.run()


@pytest.mark.linalg_qr_out
@pytest.mark.parametrize("mode", QR_MODES)
def test_linalg_qr_out(mode):
    bench = QRGenericBenchmark(
        op_name="linalg_qr_out",
        input_fn=_make_qr_out_input_fn(mode),
        torch_op=torch.linalg.qr,
        gems_op=_gems_qr_out,
        dtypes=QR_DTYPES,
    )
    bench.run()
