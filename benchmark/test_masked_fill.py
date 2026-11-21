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

from . import base, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    mask = utils.generate_tensor_input(shape, dtype, device) < 0.3
    value = 1024

    yield inp, mask, value


@pytest.mark.masked_fill
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_masked_fill():
    bench = base.GenericBenchmark(
        op_name="masked_fill", input_fn=_input_fn, torch_op=torch.masked_fill
    )
    bench.run()


@pytest.mark.masked_fill_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_masked_fill_inplace():
    bench = base.GenericBenchmark(
        op_name="masked_fill_",
        input_fn=_input_fn,
        torch_op=lambda a, b, c: a.masked_fill_(b, c),
        is_inplace=True,
    )

    bench.run()


@pytest.mark.masked_fill_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_masked_fill_scalar():
    bench = base.GenericBenchmark(
        op_name="masked_fill_scalar", input_fn=_input_fn, torch_op=torch.masked_fill
    )
    bench.run()


@pytest.mark.masked_fill_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_masked_fill_scalar_inplace():
    bench = base.GenericBenchmark(
        op_name="masked_fill_scalar_",
        input_fn=_input_fn,
        torch_op=lambda a, b, c: a.masked_fill_(b, c),
        is_inplace=True,
    )
    bench.run()


class HygonMaskedFillBenchmark(base.Benchmark):
    """Frozen representatives: three eligible sizes and three fallback routes."""

    def get_input_iter(self, dtype):
        for shape in [(1024,), (32, 64), (1 << 20,)]:
            n = 1
            for extent in shape:
                n *= extent
            inp = torch.arange(n, dtype=dtype, device=self.device).reshape(shape)
            mask = (torch.arange(n, device=self.device) % 3 == 0).reshape(shape)
            yield inp, mask, 0.5
        inp = torch.arange(2048, dtype=dtype, device=self.device).reshape(32, 64)
        mask = torch.arange(64, device=self.device) % 3 == 0
        yield inp, mask, 0.5  # Broadcast fallback.
        yield inp.t(), mask.expand(32, 64).t(), 0.5  # Strided fallback.
        yield inp, mask.expand(32, 64).contiguous(), torch.tensor(
            0.5, dtype=dtype, device=self.device
        )  # Tensor-value fallback (includes the generic value.item() sync).


@pytest.mark.masked_fill
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon fast path")
def test_masked_fill_hygon_frozen():
    bench = HygonMaskedFillBenchmark(
        op_name="masked_fill",
        torch_op=torch.masked_fill,
        gems_op=flag_gems.masked_fill,
        dtypes=[torch.float32],
    )
    bench.run()
