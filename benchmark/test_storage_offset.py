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

from . import base, consts, utils

# aten::storage_offset(Tensor self) -> int is a host-side metadata query, so the
# cases vary the view chain (the only thing that changes the reported value)
# rather than a data payload; the shared default shapes provide the sizes.
_LAYOUTS = (
    "contiguous",
    "as_strided",
    "slice_rows",
    "chained_rows",
    "select_row",
    "transpose_rows",
)


def _layout_supported(shape, layout):
    """Whether ``layout`` is a non-degenerate view of ``shape``."""
    ndim = len(shape)
    if layout == "slice_rows":
        return ndim >= 1
    if layout == "chained_rows":
        return ndim >= 1
    if layout == "select_row":
        return ndim >= 2 and shape[0] >= 2
    if layout == "transpose_rows":
        return ndim >= 2
    return True  # contiguous and as_strided are valid for every shape


def _apply_layout(base_inp, layout):
    if layout == "contiguous":
        return base_inp
    if layout == "as_strided":
        flat = base_inp.reshape(-1)
        start = min(3, flat.numel())
        return torch.as_strided(flat, (flat.numel() - start,), (1,), start)
    if layout == "slice_rows":
        return base_inp[2:]
    if layout == "chained_rows":
        return base_inp[3:][1:]
    if layout == "select_row":
        return base_inp[1]
    if layout == "transpose_rows":
        return base_inp.transpose(-1, -2)[1:]
    raise ValueError(f"unknown layout: {layout!r}")


def _case_fn(shape, dtype):
    del dtype
    for layout in _LAYOUTS:
        if not _layout_supported(shape, layout):
            continue
        yield base.BenchmarkCasePlan(
            shape={"base": shape},
            params={"layout": layout},
            builder_args=(shape, layout),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return _apply_layout(inp, layout), {}


@pytest.mark.storage_offset
def test_storage_offset():
    bench = base.GenericBenchmark(
        op_name="storage_offset",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.storage_offset,
        # The public candidate comes from the process-local override; the getattr
        # keeps case listing working before it exists.
        gems_op=getattr(flag_gems, "storage_offset", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
