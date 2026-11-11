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

import json
from pathlib import Path

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.mm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon GEMM regression coverage"
    ),
]

_DOCS = Path(__file__).resolve().parents[1] / "docs/hygon-optimization/mm"
_TARGET_ROWS = {
    case["row"]
    for case in json.loads((_DOCS / "cases.json").read_text(encoding="utf-8"))
}
_CONTROL_CASES = [
    case
    for case in json.loads((_DOCS / "all-mm-cases.json").read_text(encoding="utf-8"))
    if case["row"] not in _TARGET_ROWS
]


@pytest.mark.parametrize("case", _CONTROL_CASES, ids=lambda case: str(case["row"]))
@pytest.mark.parametrize("seed", [100003, 880301])
def test_hygon_mm_spreadsheet_controls_public(case, seed):
    operands = []
    cpu_operands = []
    for index, meta in enumerate(case["metadata"][0]):
        dtype = getattr(torch, meta["dtype"].split(".")[-1])
        offset = meta.get("storage_offset", 0)
        size = (
            offset
            + 1
            + sum((n - 1) * stride for n, stride in zip(meta["shape"], meta["stride"]))
        )
        generator = torch.Generator().manual_seed(seed + case["row"] * 2 + index)
        storage = torch.randn(size, dtype=dtype, generator=generator)
        cpu_operands.append(storage.as_strided(meta["shape"], meta["stride"], offset))
        operand = storage.to(flag_gems.device).as_strided(
            meta["shape"], meta["stride"], offset
        )
        operand.requires_grad_(meta.get("requires_grad", False))
        operands.append(operand)
    reference = cpu_operands[0].double() @ cpu_operands[1].double()
    dtype = operands[0].dtype
    if dtype == torch.bfloat16:
        reference = reference.to(dtype).double()
        atol, rtol = 1e-5, 0.016
    elif dtype == torch.float64:
        atol, rtol = 1e-10, 1e-10
    else:
        atol, rtol = 2e-4, 1e-4
    with flag_gems.use_gems(include=["mm"]):
        output = torch.mm(*operands)
    assert output.dtype == dtype and output.device == operands[0].device
    torch.testing.assert_close(output.cpu().double(), reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seed", [880301, 7, 101])
@pytest.mark.parametrize("offset", [0, 17])
def test_hygon_mm_bf16_small_public(seed, offset):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((32, 64), dtype=torch.bfloat16, generator=gen)
    b = torch.randn((64, 32), dtype=torch.bfloat16, generator=gen)
    aa = a.to(flag_gems.device)
    storage = torch.empty(offset + b.numel(), dtype=b.dtype, device=flag_gems.device)
    bb = storage.as_strided(b.shape, b.stride(), offset)
    bb.copy_(b)
    with flag_gems.use_gems(include=["mm"]):
        result = torch.mm(aa, bb).cpu()
    torch.testing.assert_close(result, a @ b, rtol=0.016, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("m,n,k", [(0, 17, 9), (13, 0, 9), (13, 17, 0)])
def test_hygon_mm_empty(dtype, m, n, k):
    a = torch.empty((m, k), dtype=dtype, device=flag_gems.device)
    b = torch.empty((k, n), dtype=dtype, device=flag_gems.device)
    y = flag_gems.mm(a, b)
    assert y.shape == (m, n)
    assert y.dtype == dtype
    if k == 0:
        assert torch.count_nonzero(y.cpu()).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [7, 101])
def test_hygon_mm_offset_and_both_strides(dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    # Offset, padded rows and a column stride greater than one on both inputs.
    a = torch.randn(41 * 137 + 11, dtype=dtype, generator=gen).as_strided(
        (33, 65), (137, 2), 7
    )
    b = torch.randn(65 * 83 + 11, dtype=dtype, generator=gen).as_strided(
        (65, 37), (83, 2), 5
    )

    # Transfer underlying storage to preserve the requested offset and strides.
    def transfer(x):
        storage = torch.as_strided(
            x, (x.untyped_storage().nbytes() // x.element_size(),), (1,), 0
        )
        return storage.to(flag_gems.device).as_strided(
            x.shape, x.stride(), x.storage_offset()
        )

    aa, bb = transfer(a), transfer(b)
    y = flag_gems.mm(aa, bb).cpu().double()
    ref = a.double() @ b.double()
    tol = 1e-10 if dtype == torch.float64 else 2e-4
    torch.testing.assert_close(y, ref, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (192, 128, 128),
        (192, 6528, 89),
        (2208, 1280, 64),
        (1344, 128, 256),
        (2208, 64, 64),
        (576, 128, 128),
        (2208, 512, 64),
        (576, 128, 384),
        (1344, 128, 128),
        (192, 2944, 89),
        (192, 512, 89),
        (960, 128, 384),
    ],
)
def test_hygon_mm_fp64_keeps_low_mantissa_bits(m, n, k):
    # These bits disappear if an MMAC operand is silently converted to FP32.
    a = torch.zeros((m, k), dtype=torch.float64)
    b = torch.ones((k, n), dtype=torch.float64)
    a[:, 0] = 1.0 + 2.0**-30
    a[:, 1] = -1.0
    y = flag_gems.mm(a.to(flag_gems.device), b.to(flag_gems.device)).cpu()
    torch.testing.assert_close(y, a @ b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1344, 128, 256),
        (2208, 64, 64),
        (576, 128, 128),
        (2208, 512, 64),
        (576, 128, 384),
    ],
)
@pytest.mark.parametrize("seed", [1238, 101241, 907])
def test_hygon_mm_fp64_reported_peel_regression(m, n, k, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.float64, generator=gen)
    b = torch.randn((k, n), dtype=torch.float64, generator=gen)
    reference = a @ b
    aa = a.to(flag_gems.device)
    outputs = []
    for offset in [0, 131072]:
        # Preserve identical logical inputs while changing only B's offset.
        storage = torch.empty(offset + b.numel(), dtype=b.dtype, device=aa.device)
        bb = storage.as_strided(b.shape, b.stride(), offset)
        bb.copy_(b)
        assert bb.storage_offset() == offset
        assert torch.equal(bb.cpu(), b)
        result = flag_gems.mm(aa, bb).cpu()
        torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10)
        outputs.append(result)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
