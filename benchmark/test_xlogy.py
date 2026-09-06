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

from . import base, consts


def xlogy_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    # keep ``other`` positive so ``log`` stays finite
    inp2 = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
    yield inp1, inp2


@pytest.mark.xlogy
def test_xlogy():
    bench = base.GenericBenchmark(
        op_name="xlogy",
        torch_op=torch.xlogy,
        input_fn=xlogy_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def xlogy_out_input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.xlogy_out
def test_xlogy_out():
    bench = base.GenericBenchmark(
        op_name="xlogy_out",
        torch_op=torch.xlogy,
        input_fn=xlogy_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def xlogy_tensor_scalar_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    # Reference proxy: the torch_xmlir reference kernel for xlogy.Scalar_Other
    # crashes with [INVALID PARAMETER] (CUDANativeFunctions.cpp:16268, NON_BUG
    # reference-side limitation), so the reference is timed through the
    # numerically identical xlogy.Tensor 0D broadcast instead.  The gems side
    # is explicitly bound to the real Scalar_Other vendor kernel below.
    yield inp, torch.tensor(3.5, dtype=dtype, device=device)


@pytest.mark.xlogy_tensor_scalar
def test_xlogy_tensor_scalar():
    bench = base.GenericBenchmark(
        op_name="xlogy_tensor_scalar",
        torch_op=torch.xlogy,
        input_fn=xlogy_tensor_scalar_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems.xlogy_tensor_scalar,
    )
    bench.run()


def xlogy_tensor_scalar_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    # Reference proxy (see xlogy_tensor_scalar_input_fn): xlogy.OutTensor 0D
    # broadcast; OutScalar_Other reference crashes.  Measured op is the real
    # OutScalar_Other vendor kernel.
    yield inp, torch.tensor(3.5, dtype=dtype, device=device), {"out": out}


@pytest.mark.xlogy_tensor_scalar_out
def test_xlogy_tensor_scalar_out():
    bench = base.GenericBenchmark(
        op_name="xlogy_tensor_scalar_out",
        torch_op=torch.xlogy,
        input_fn=xlogy_tensor_scalar_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems.xlogy_tensor_scalar_out,
    )
    bench.run()


def xlogy_scalar_tensor_input_fn(shape, dtype, device):
    # keep ``other`` positive so ``log`` stays finite
    inp = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
    # Reference proxy (see xlogy_tensor_scalar_input_fn): xlogy.Tensor 0D
    # broadcast; Scalar_Self reference crashes.  Measured op is the real
    # Scalar_Self vendor kernel.
    yield torch.tensor(2.0, dtype=dtype, device=device), inp


@pytest.mark.xlogy_scalar_tensor
def test_xlogy_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="xlogy_scalar_tensor",
        torch_op=torch.xlogy,
        input_fn=xlogy_scalar_tensor_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems.xlogy_scalar_tensor,
    )
    bench.run()


def xlogy_scalar_tensor_out_input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) * 5.0 + 0.01
    out = torch.empty(shape, dtype=dtype, device=device)
    # Reference proxy (see xlogy_tensor_scalar_input_fn): xlogy.OutTensor 0D
    # broadcast; OutScalar_Self reference crashes.  Measured op is the real
    # OutScalar_Self vendor kernel.
    yield torch.tensor(2.0, dtype=dtype, device=device), inp, {"out": out}


@pytest.mark.xlogy_scalar_tensor_out
def test_xlogy_scalar_tensor_out():
    bench = base.GenericBenchmark(
        op_name="xlogy_scalar_tensor_out",
        torch_op=torch.xlogy,
        input_fn=xlogy_scalar_tensor_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems.xlogy_scalar_tensor_out,
    )
    bench.run()
