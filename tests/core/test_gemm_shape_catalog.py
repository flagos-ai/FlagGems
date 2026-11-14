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

"""The platform core suite has a fixed, tensor-free case contract."""

import pytest

from benchmark.gemm_shapes import _catalog, accuracy_cases, blas_cases, gemm_layouts


@pytest.mark.parametrize(
    "op,count",
    [
        ("mm", 62),
        ("addmm", 62),
        ("mv", 22),
        ("router_gemm", 24),
        ("bmm", 45),
        ("baddbmm", 45),
    ],
)
def test_platform_core_case_counts(op, count):
    shapes = _catalog()[op]["shapes"]
    cases = (
        [(tuple(s), layout) for s in shapes for layout in gemm_layouts(op, s)]
        if op == "mv"
        else list(blas_cases(op, shapes))
    )
    assert len(cases) == len(set(cases)) == count
    assert list(accuracy_cases(op))
    assert list(accuracy_cases(op, quick=True))


def test_shared_operator_shape_sets():
    catalog = _catalog()
    assert catalog["mm"]["shapes"] == catalog["addmm"]["shapes"]
    assert catalog["bmm"]["shapes"] == catalog["baddbmm"]["shapes"]
    assert {s[0] for s in catalog["mm"]["shapes"]} == {1}
    assert {s[0] for s in catalog["bmm"]["shapes"]} == {2, 4, 8}


def test_explicit_layout_selection_and_other_blas_order():
    shape = (1, 32, 512, 4096)
    assert list(blas_cases("mm", [shape], requested="nt")) == [(*shape, True)]
    assert gemm_layouts("mm", (1, 64, 1, 4096)) == (False,)
    assert gemm_layouts("mm", (1, 64, 1, 4096), requested="both") == (False, True)
    assert gemm_layouts("bmm", shape, requested="nn") == (False, True)
    assert gemm_layouts("mv", (1, 4096)) == (False,)
    shapes = [shape, (1, 64, 64, 2048)]
    assert list(blas_cases("other", shapes, comprehensive=True)) == [
        (*s, layout) for layout in (False, True) for s in shapes
    ]
