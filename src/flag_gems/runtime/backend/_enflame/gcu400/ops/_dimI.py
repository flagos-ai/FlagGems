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

import logging

import torch

logger = logging.getLogger(__name__)

# GCU400 没有 SparseCOO TensorImpl，aten 的 sparse COO 构造算子未注册到
# SparsePrivateUse1，导致 sparse 元数据算子的测试在构造输入时抛
# NotImplementedError。这里补一份 CPU 侧构造实现：元数据查询不读取存储数据，
# 把 indices/values 复制到 CPU 后构造等价 COO 张量，sparse_dim / dense_dim /
# shape / coalesced 状态保持不变。注册幂等，_dimV 等同类算子可复用。
_SPARSE_COO_FALLBACK_FLAG = "_flag_gems_gcu400_sparse_coo_fallback_registered"
_SPARSE_COO_LIBRARY = None


def ensure_sparse_coo_gcu_fallback():
    """幂等注册 SparsePrivateUse1 的 sparse COO 构造实现，供 sparse 元数据算子复用。"""
    global _SPARSE_COO_LIBRARY
    if getattr(torch, _SPARSE_COO_FALLBACK_FLAG, False):
        return

    def build_cpu_sparse_coo(
        sparse_dim,
        dense_dim,
        size,
        indices,
        values,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=False,
        is_coalesced=None,
    ):
        # sparse_dim / dense_dim / size 由 indices 与 values 的形状隐式给出；
        # device 固定为 CPU，layout 固定为 sparse_coo，其余参数与 aten schema 对齐。
        cpu_indices = indices.detach().to("cpu")
        cpu_values = values.detach().to("cpu")
        if dtype is not None:
            cpu_values = cpu_values.to(dtype)
        options = {}
        if is_coalesced is not None:
            options["is_coalesced"] = is_coalesced
        return torch.sparse_coo_tensor(cpu_indices, cpu_values, tuple(size), **options)

    _SPARSE_COO_LIBRARY = torch.library.Library("aten", "IMPL")
    try:
        _SPARSE_COO_LIBRARY.impl(
            "_sparse_coo_tensor_with_dims_and_tensors",
            build_cpu_sparse_coo,
            "SparsePrivateUse1",
        )
    except RuntimeError as error:
        logger.warning("sparse COO gcu400 fallback registration skipped: %s", error)
    setattr(torch, _SPARSE_COO_FALLBACK_FLAG, True)


ensure_sparse_coo_gcu_fallback()


_COO_LAYOUT = torch.sparse_coo


def _dimI(self: torch.Tensor) -> int:
    """Return the number of sparse dimensions of a sparse COO tensor ``self``.

    Mirrors ``aten::_dimI(Tensor self) -> int``. This is a pure layout-metadata
    query: the answer is a property of the tensor's index layout, never of the
    stored element data, so there is no data movement and no Triton kernel.
    The result is read from TensorImpl/sparse-COO metadata without
    re-dispatching the ``_dimI`` operator itself (re-entering the dispatcher
    from inside a kernel registered on the Sparse dispatch key would recurse
    into this implementation).

    Semantics by layout (verified against native ATen on H20, torch
    2.9.0+cu128):
    - ``torch.sparse_coo``: ``_dimI`` is the sparse dimension count, i.e. the
      leading dimension of the index matrix, ``self._indices().shape[0]``
      (the native COO TensorImpl keeps the index matrix at shape
      ``(sparse_dim, nnz)``, so this equals ``self.sparse_dim()``). Plain 2-D
      COO gives 2, fully sparse rank-N COO gives N, and hybrid COO tensors
      (values carrying trailing dense dimensions) give the number of sparse
      dims only. Zero-nnz and uncoalesced tensors report the same value as
      their coalesced counterparts, because coalescing state does not change
      the layout.
    - Every other layout: native ATen registers ``aten::_dimI`` only for the
      Sparse (COO) backends -- strided, CSR/CSC (and BSR/BSC) raise
      ``NotImplementedError`` ("Could not run 'aten::_dimI' ... from the
      'CUDA'/'SparseCsrCUDA' backend"). Mirror that contract here instead of
      guessing a formula.
    """
    logger.debug("GEMS _DIMI")
    if self.layout is torch.sparse_coo:
        return int(self._indices().shape[0])
    raise NotImplementedError(
        f"Could not run 'aten::_dimI' with arguments from the '{self.layout}' "
        "layout; aten::_dimI is only registered for the Sparse (COO) backends"
    )
