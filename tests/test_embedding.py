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

import os
import random
import time

import pytest
import torch

import flag_gems
from flag_gems.ops.embedding import embedding as generic_embedding
from flag_gems.ops.embedding import embedding_backward as generic_embedding_backward

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

device = flag_gems.device


def _gcu_int64_runtime_enabled():
    """True when the GCU 64-bit switch keeps int64 tensors on device.

    ``flag_gems.runtime.device.support_int64`` only reports a static per-arch
    capability (true for every GCU400 build), while the runtime behaviour is
    driven by ``TORCH_GCU_ENABLE_INT64_AND_UINT64``: with the switch off, GCU
    silently narrows int64 tensors to int32 before the reference runs, so the
    native embedding backward reference accepts the int64 index tensors these
    tests build. With the switch on only the int32 index cases stay runnable.
    """
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1"


@pytest.mark.embedding
@pytest.mark.parametrize(
    "EmbeddingSize", [1024] if cfg.TO_CPU or cfg.QUICK_MODE else [4096]
)
@pytest.mark.parametrize("Batch", [2] if cfg.TO_CPU or cfg.QUICK_MODE else [2, 4])
@pytest.mark.parametrize("M", [4] if cfg.TO_CPU or cfg.QUICK_MODE else [4, 8])
@pytest.mark.parametrize("N", [8] if cfg.TO_CPU or cfg.QUICK_MODE else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [None] if cfg.QUICK_MODE else [None, -1, 1, 2])
@pytest.mark.parametrize(
    "scale_grad_by_freq", [False] if cfg.QUICK_MODE else [True, False]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device=flag_gems.device, requires_grad=False
    )
    res_embedding = torch.randn(
        (EmbeddingSize, N), device=flag_gems.device, dtype=dtype, requires_grad=True
    )
    ref_embedding = utils.to_reference(res_embedding)
    ref_indices = utils.to_reference(res_indices)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    res_out = flag_gems.embedding(
        res_embedding,
        res_indices,
        padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.embedding
@pytest.mark.skipif(
    flag_gems.embedding is not generic_embedding,
    reason="Requires the generic embedding implementation",
)
@pytest.mark.parametrize(
    "index_data,width,padding_idx",
    [
        pytest.param([0, 0, 2], 4, -1, id="duplicate-indices"),
        pytest.param([[0, 5], [2, 0]], 33, 0, id="padding-lookup"),
        pytest.param(2, 1, -1, id="scalar-index"),
        pytest.param([], 7, -1, id="empty-indices"),
        pytest.param([0, 2], 0, -1, id="zero-width"),
    ],
)
def test_embedding_sparse_forward(index_data, width, padding_idx):
    weight = torch.arange(6 * width, dtype=torch.float32).reshape(6, width)
    indices = torch.tensor(index_data, dtype=torch.int64)
    expected = torch.ops.aten.embedding.default(
        weight, indices, padding_idx, False, True
    )
    actual = flag_gems.embedding(
        weight.to(device), indices.to(device), padding_idx, sparse=True
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _check_sparse_embedding_backward(indices, width, padding_idx):
    grad = torch.arange(indices.numel() * width, dtype=torch.float32).reshape(
        *indices.shape, width
    )
    expected = torch.ops.aten.embedding_backward.default(
        grad, indices, 6, padding_idx, False, True
    )
    actual = flag_gems.embedding_backward(
        grad.to(device), indices.to(device), 6, padding_idx, sparse=True
    )

    assert actual.layout == torch.sparse_coo
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.sparse_dim() == expected.sparse_dim()
    assert actual.dense_dim() == expected.dense_dim()
    assert actual._nnz() == expected._nnz()
    # A dense comparison would miss zero-valued padding entries in the COO.
    torch.testing.assert_close(
        actual._indices().cpu(), expected._indices(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual._values().cpu(), expected._values(), rtol=0, atol=0
    )


@pytest.mark.embedding_backward
@pytest.mark.skipif(
    flag_gems.embedding_backward is not generic_embedding_backward,
    reason="Requires the generic embedding backward implementation",
)
@pytest.mark.parametrize(
    "index_data,index_dtype,width,padding_idx",
    [
        pytest.param([0, 0, 2], torch.int64, 4, -1, id="duplicate-indices"),
        pytest.param([0, 2, 0, 2, 5, 0], torch.int32, 33, 0, id="mixed-padding"),
        pytest.param([5, 5], torch.int64, 4, 5, id="all-padding"),
        pytest.param([4, 1, 2, 1], torch.int64, 1, 0, id="padding-absent"),
        pytest.param(2, torch.int64, 7, -1, id="scalar-index"),
        pytest.param([], torch.int32, 4, 0, id="empty-indices"),
        pytest.param([0, 5, 2], torch.int64, 0, 0, id="zero-width"),
    ],
)
def test_embedding_sparse_gradient(index_data, index_dtype, width, padding_idx):
    _check_sparse_embedding_backward(
        torch.tensor(index_data, dtype=index_dtype), width, padding_idx
    )


@pytest.mark.embedding_backward
@pytest.mark.skipif(
    flag_gems.embedding_backward is not generic_embedding_backward,
    reason="Requires the generic embedding backward implementation",
)
@pytest.mark.parametrize("rows", [255, 256, 257, 16385])
def test_embedding_sparse_gradient_scan(rows):
    _check_sparse_embedding_backward(torch.arange(rows, dtype=torch.int32) % 6, 9, 0)


@pytest.mark.embedding_backward
@pytest.mark.skipif(
    flag_gems.embedding_backward is not generic_embedding_backward,
    reason="Requires the generic embedding backward implementation",
)
def test_embedding_sparse_gradient_rejects_frequency_scaling():
    grad = torch.ones(3, 4, dtype=torch.float32)
    indices = torch.tensor([0, 0, 2], dtype=torch.int64)
    message = "scale_grad_by_freq not supported with sparse gradients"
    with pytest.raises(RuntimeError, match=message):
        torch.ops.aten.embedding_backward.default(grad, indices, 6, -1, True, True)
    with pytest.raises(RuntimeError, match=message):
        flag_gems.embedding_backward(
            grad.to(device), indices.to(device), 6, -1, True, True
        )


@pytest.mark.embedding_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame"
    and _gcu_int64_runtime_enabled()
    and not cfg.TO_CPU,
    reason="native reference aten::embedding_backward (topsatenEmbeddingBackward) "
    "only accepts int32 index tensors and aborts the process with "
    "TOPSATEN_STATUS_BAD_PARAM for the int64 index tensor built by this test when "
    "the TORCH_GCU_ENABLE_INT64_AND_UINT64 switch is on; with the switch off GCU "
    "narrows the index to int32 before dispatch and with --ref cpu the reference "
    "runs on CPU, so only the int64-enabled device-reference run is skipped",
)
@pytest.mark.parametrize(
    "EmbeddingSize", [1024] if cfg.TO_CPU or cfg.QUICK_MODE else [4096]
)
@pytest.mark.parametrize("Batch", [2] if cfg.TO_CPU or cfg.QUICK_MODE else [2, 4])
@pytest.mark.parametrize("M", [4] if cfg.TO_CPU or cfg.QUICK_MODE else [4, 8])
@pytest.mark.parametrize("N", [8] if cfg.TO_CPU or cfg.QUICK_MODE else [128, 256, 4096])
@pytest.mark.parametrize("padding_idx", [-1] if cfg.QUICK_MODE else [-1, 1, 2])
@pytest.mark.parametrize(
    "scale_grad_by_freq", [False] if cfg.QUICK_MODE else [True, False]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_backward(
    EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype
):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_grad = torch.randn((Batch, M, N), device=flag_gems.device, dtype=dtype)
    res_indices = torch.randint(0, EmbeddingSize, (Batch, M), device=flag_gems.device)
    num_weights = EmbeddingSize
    sparse = False

    ref_grad = utils.to_reference(res_grad)
    ref_indices = utils.to_reference(res_indices)

    ref_in_grad = torch.ops.aten.embedding_backward(
        ref_grad, ref_indices, num_weights, padding_idx, scale_grad_by_freq, sparse
    )
    res_in_grad = flag_gems.embedding_backward(
        res_grad, res_indices, num_weights, padding_idx, scale_grad_by_freq, sparse
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.embedding_dense_backward
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.parametrize(
    "Batch, M, N, embeddingsize",
    [
        (2, 4, 8, 16),
        (4, 8, 32, 64),
        (1, 3, 64, 128),
    ],
)
@pytest.mark.parametrize(
    "padding_idx, scale_grad_by_freq", [(-1, False), (0, True), (5, False)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("seed", [42])
def test_embedding_dense_backward(
    Batch, M, N, embeddingsize, padding_idx, scale_grad_by_freq, dtype, seed
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    grad_output = torch.randn((Batch, M, N), device=flag_gems.device, dtype=dtype)
    indices = torch.randint(
        0, embeddingsize, (Batch, M), device=flag_gems.device, dtype=torch.long
    )

    if padding_idx >= 0 and embeddingsize > 0:
        mask = torch.rand((Batch, M), device=flag_gems.device) < 0.25
        indices = torch.where(mask, torch.full_like(indices, padding_idx), indices)
    num_weights = embeddingsize
    ref_grad_output = utils.to_reference(grad_output)
    ref_indices = utils.to_reference(indices)
    if flag_gems.vendor_name == "enflame":
        # torch_gcu routes this op to topsaten, whose kernel contract requires
        # int32 indices and aborts the process on int64. The result is
        # index-dtype independent, so only the reference input is downcast.
        ref_indices = ref_indices.to(torch.int32)
    ref_out = torch.ops.aten.embedding_dense_backward(
        ref_grad_output,
        ref_indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
    )

    res_out = flag_gems.embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )
    # res_out = torch.ops.aten.embedding_dense_backward(
    # grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)

    utils.gems_assert_close(res_out, ref_out, dtype)
