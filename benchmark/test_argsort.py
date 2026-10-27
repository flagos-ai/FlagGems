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


class ArgsortBenchmark(base.GenericBenchmark2DOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_failures = []

    def _measure_input(self, input, case_id=None, dtype=None):
        # Complete timing independently before checking this exact input shape.
        metric = super()._measure_input(input, case_id=case_id, dtype=dtype)
        args, kwargs = self.unpack_to_args_kwargs(input)
        inp = args[0]
        try:
            actual = self.gems_op(*args, **kwargs).cpu()
            reference = torch.argsort(inp.cpu(), stable=True, **kwargs)
            correct = (
                actual.dtype == torch.int64
                and actual.shape == reference.shape
                and torch.equal(actual, reference)
            )
            metric.accuracy = float(correct)
            if not correct:
                raise AssertionError("stable argsort indices differ from CPU reference")
        except Exception as exc:
            metric.accuracy = 0.0
            failure = f"dtype={inp.dtype}, shape={tuple(inp.shape)}: {exc}"
            metric.error_msg = f"Accuracy check failed: {failure}"
            self.accuracy_failures.append(failure)
        return metric

    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (65536, 4), (98307, 8)]


def _input_fn(shape, dtype, device):
    if dtype in (torch.int8, torch.uint8):
        low, high = (-128, 128) if dtype == torch.int8 else (0, 256)
        inp = torch.randint(low, high, shape, dtype=dtype, device="cpu").to(device)
    elif dtype == torch.int64:
        inp = torch.randint(-(2**60), 2**60, shape, dtype=dtype, device="cpu").to(
            device
        )
    else:
        inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1, "descending": False},


@pytest.mark.argsort
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            dtype,
            marks=pytest.mark.skip_native(
                vendors=["ascend"],
                reason="Native integer argsort falls back to AiCPU",
            ),
        )
        for dtype in consts.INT_DTYPES + consts.EXTRA_INT_DTYPES
    ]
    + consts.FLOAT_DTYPES,
)
def test_argsort(dtype):
    bench = ArgsortBenchmark(
        input_fn=_input_fn,
        op_name="argsort",
        torch_op=torch.argsort,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems.argsort)
    bench.run()
    assert not bench.accuracy_failures, "\n".join(bench.accuracy_failures)
