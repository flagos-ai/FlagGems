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

# ``_has_compatible_shallow_copy_type`` is a metadata predicate over
# DispatchKeySets, so accuracy here is not about numerics but about covering
# every TensorImpl family a key set can encode. The tests below build one tensor
# per family and compare the gems result against the native one for *all*
# ordered pairs, which is the only way to catch a decision rule that agrees on
# dense inputs while silently mis-classifying meta, MKL-DNN, nested, quantized
# or sparse-compressed tensors.

_CUDA = torch.cuda.is_available() and flag_gems.device == "cuda"


def _tensor_kinds():
    """One representative tensor per TensorImpl family, keyed by a short name."""
    base = torch.randn(4, 4)
    kinds = {
        "dense": base,
        "dense_t": base.t(),
        "dense_view": base.view(16),
        "dense_slice": base[1:3],
        "dense_expand": torch.randn(1, 4).expand(4, 4),
        "dense_scalar": torch.tensor(1.0),
        "dense_empty": torch.empty(0),
        "dense_int": torch.ones(4, 4, dtype=torch.int32),
        "dense_bool": torch.ones(4, 4, dtype=torch.bool),
        "dense_half": torch.ones(4, 4, dtype=torch.float16),
        "dense_complex": torch.ones(4, 4, dtype=torch.complex64),
        "dense_channels_last": torch.randn(1, 3, 4, 4).to(
            memory_format=torch.channels_last
        ),
        "dense_requires_grad": torch.randn(4, 4, requires_grad=True),
        "meta": torch.randn(4, 4, device="meta"),
        "mkldnn": base.to_mkldnn(),
        "nested": torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(3, 3)]),
        "quantized": torch.quantize_per_tensor(base, 0.1, 0, torch.quint8),
        "sparse_coo": base.to_sparse(),
        "sparse_coo_coalesced": base.to_sparse().coalesce(),
        "sparse_csr": base.to_sparse_csr(),
        "sparse_csc": base.to_sparse_csc(),
        "sparse_bsr": base.to_sparse_bsr((2, 2)),
        "sparse_bsc": base.to_sparse_bsc((2, 2)),
    }
    if _CUDA:
        dev = flag_gems.device
        kinds["dense_cuda"] = base.to(dev)
        kinds["dense_cuda_requires_grad"] = torch.randn(
            4, 4, device=dev, requires_grad=True
        )
        kinds["sparse_coo_cuda"] = base.to_sparse().to(dev)
        kinds["sparse_csr_cuda"] = base.to_sparse_csr().to(dev)
    return kinds


TENSOR_KINDS = _tensor_kinds()
KIND_NAMES = sorted(TENSOR_KINDS)
DENSE_NAMES = [name for name in KIND_NAMES if name.startswith("dense")]
COO_NAMES = [name for name in KIND_NAMES if name.startswith("sparse_coo")]
COMPRESSED_NAMES = [
    name
    for name in KIND_NAMES
    if name.startswith("sparse_") and not name.startswith("sparse_coo")
]


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize("self_name", KIND_NAMES)
def test_accuracy_has_compatible_shallow_copy_type(self_name):
    # Differential test: for this ``self`` kind, the gems result must match the
    # native result against every other kind. The result is a bool, so no
    # device or dtype normalisation of the reference is needed.
    self_tensor = TENSOR_KINDS[self_name]

    mismatches = []
    for from_name in KIND_NAMES:
        from_tensor = TENSOR_KINDS[from_name]

        ref_out = torch._has_compatible_shallow_copy_type(self_tensor, from_tensor)
        with flag_gems.use_gems():
            res_out = torch._has_compatible_shallow_copy_type(self_tensor, from_tensor)

        assert isinstance(res_out, bool)
        if res_out != ref_out:
            mismatches.append((from_name, ref_out, res_out))

    assert not mismatches, "self={}: {}".format(
        self_name,
        ", ".join(f"from={f} ref={r} res={g}" for f, r, g in mismatches),
    )


@pytest.mark.has_compatible_shallow_copy_type
def test_accuracy_has_compatible_shallow_copy_type_dense_family():
    # Every dense tensor is compatible with every other dense tensor: shape,
    # dtype, stride, memory format and device are all irrelevant.
    for self_name in DENSE_NAMES:
        for from_name in DENSE_NAMES:
            with flag_gems.use_gems():
                res_out = torch._has_compatible_shallow_copy_type(
                    TENSOR_KINDS[self_name], TENSOR_KINDS[from_name]
                )
            assert res_out is True, f"{self_name} vs {from_name}"


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize("family", ["coo", "compressed"])
def test_accuracy_has_compatible_shallow_copy_type_sparse_family(family):
    # Sparse COO is closed under itself, and the four compressed layouts are
    # closed under each other. Both hold across devices.
    names = COO_NAMES if family == "coo" else COMPRESSED_NAMES
    for self_name in names:
        for from_name in names:
            with flag_gems.use_gems():
                res_out = torch._has_compatible_shallow_copy_type(
                    TENSOR_KINDS[self_name], TENSOR_KINDS[from_name]
                )
            assert res_out is True, f"{self_name} vs {from_name}"


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize("self_name", ["meta", "mkldnn", "nested", "quantized"])
def test_accuracy_has_compatible_shallow_copy_type_opaque(self_name):
    # Opaque impls live on their own backends and are only compatible with a
    # tensor carrying an identical key set. They must not be treated as dense,
    # even though meta and nested tensors report ``torch.strided`` as layout.
    for from_name in KIND_NAMES:
        expected = from_name == self_name
        with flag_gems.use_gems():
            res_out = torch._has_compatible_shallow_copy_type(
                TENSOR_KINDS[self_name], TENSOR_KINDS[from_name]
            )
        assert res_out is expected, f"{self_name} vs {from_name}"


@pytest.mark.has_compatible_shallow_copy_type
@pytest.mark.parametrize(
    "self_name, from_name",
    [
        ("dense", "sparse_coo"),
        ("dense", "sparse_csr"),
        ("sparse_coo", "dense"),
        ("sparse_coo", "sparse_csr"),
        ("sparse_csr", "sparse_coo"),
        ("dense", "meta"),
        ("dense", "nested"),
        ("dense", "quantized"),
    ],
)
def test_accuracy_has_compatible_shallow_copy_type_cross_family(self_name, from_name):
    # Different families are never shallow-copy compatible.
    ref_out = torch._has_compatible_shallow_copy_type(
        TENSOR_KINDS[self_name], TENSOR_KINDS[from_name]
    )
    with flag_gems.use_gems():
        res_out = torch._has_compatible_shallow_copy_type(
            TENSOR_KINDS[self_name], TENSOR_KINDS[from_name]
        )

    assert res_out == ref_out
    assert res_out is False
