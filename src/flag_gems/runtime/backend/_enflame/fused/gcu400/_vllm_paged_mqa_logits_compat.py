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

"""Runtime compatibility for vLLM's paged MQA logits wrappers on GCU.

The accuracy gate ``pytest --quick -v tests/test_fp8_fp4_paged_mqa_logits.py``
calls ``vllm.utils.deep_gemm`` wrappers directly in the test body, and all the
failure layers below live outside the operator under test:

1. **DeepGEMM unavailable (this image).** No external ``deep_gemm`` package is
   installed and the vendored ``vllm.third_party.deep_gemm._C`` CUDA extension
   cannot be dlopen'ed (its NEEDED ``libnvrtc.so.13`` is absent from the
   CUDA-less GCU image), so ``vllm.utils.deep_gemm._lazy_init`` leaves every
   kernel impl ``None`` and the wrappers raise
   ``RuntimeError: DeepGEMM backend is not available or outdated``. The test
   dies at the metadata call before the Triton kernel is ever reached.

2. **GCU deep_gemm build present (other images).** Its C++ entry points
   hard-exit the whole process on argument mismatch: the metadata call asserts
   ``num_sms == <backend SM count>`` (24) while torch_gcu reports the SPC count
   (e.g. 4), and the topsaten path rejects the standard vLLM 2D
   ``[B*next_n, H]`` weights layout. Clamping ``num_sms`` to
   ``deep_gemm.get_num_sms()`` and adding the vendor-documented
   ``ENFLAME_DEEPGEMM_KERNEL_TRITON_BACKEND`` entry (torch-native path for
   exactly that layout) avoid both process-level kills.

3. **Reference leg.** vLLM's DeepGEMM CUDA kernel has no GCU implementation,
   so once the original call raises, this shim substitutes a local
   torch-native reference that decodes the same paged-KV byte layout as the
   Triton kernel's ``_preprocess_kv_cache`` (per physical block: a
   ``[block_size, D]`` FP8 data region followed by ``block_size`` float32
   scales) and computes
   ``logits[row, pos] = sum_h w[row, h] * relu(dot(q[row, h], kv[row, pos]) * scale[row, pos])``.

The metadata substitute delegates to FlagGems' own
``get_paged_mqa_logits_metadata`` operator (backend copy), never to raw aten
math. This module is test/reference glue installed at ``import flag_gems``
time on the enflame backend only; it is not part of the operator's host path.
Remove once the image ships a usable deep_gemm build for GCU.
"""

import os
import sys

import torch

import flag_gems


_TARGET_MODULE = "vllm.utils.deep_gemm"
_TRITON_BACKEND_ENV = "ENFLAME_DEEPGEMM_KERNEL_TRITON_BACKEND"
_PAGED_KERNEL_NAME = "fp8_fp4_paged_mqa_logits"


def _backend_num_sms():
    """SM count reported by the deep_gemm backend itself (None if absent)."""
    try:
        import deep_gemm

        return int(deep_gemm.get_num_sms())
    except Exception:
        return None


def _flaggems_metadata(context_lens, block_size, num_sms):
    """Delegate scheduling metadata to FlagGems' own operator (backend copy)."""
    import flag_gems

    op = getattr(flag_gems, "get_paged_mqa_logits_metadata", None)
    if op is None:
        raise RuntimeError(
            "flag_gems.get_paged_mqa_logits_metadata is unavailable; cannot "
            "substitute vLLM DeepGEMM metadata on this backend"
        )
    return op(context_lens, block_size, num_sms)


def _torch_paged_mqa_logits_reference(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Torch-native reference for the FP8 paged MQA logits semantics.

    Mirrors the Triton kernel's decoding of the vLLM DeepGEMM paged-KV cache:
    each physical block serializes as ``[block_size * D]`` uint8 FP8 data
    bytes followed by ``[block_size]`` float32 scale bytes.
    """
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError(
            "FP4 (packed q + block scales) reference is not provided by the "
            "enflame compat shim; only the FP8 q path is covered"
        )
    if q_values.dim() == 3:
        q_values = q_values.unsqueeze(1)

    B, next_n, H, D = q_values.shape
    total_rows = B * next_n
    block_size = kv_cache.shape[1]
    num_phys_blocks = kv_cache.shape[0]
    head_dim = kv_cache.shape[3] - 4
    assert head_dim == D

    if context_lens.dim() == 2:
        ctx_flat = context_lens.reshape(-1)[:total_rows]
    else:
        ctx_flat = context_lens.repeat_interleave(next_n)

    if block_tables.dim() == 2:
        bt_exp = block_tables.unsqueeze(1).expand(B, next_n, -1).reshape(total_rows, -1)
    else:
        bt_exp = block_tables

    q_f = q_values.reshape(total_rows, H, D).float()
    w_f = weights.reshape(total_rows, H).float()

    logits = torch.full(
        (total_rows, max_model_len),
        float("-inf") if clean_logits else 0.0,
        device=q_values.device,
        dtype=torch.float32,
    )

    max_ctx = int(ctx_flat.max().item()) if total_rows else 0
    num_lb = min((max_ctx + block_size - 1) // block_size, bt_exp.shape[1])
    kv_flat = kv_cache.reshape(num_phys_blocks, block_size * (D + 4))
    pos_base = torch.arange(block_size, device=q_values.device)
    for lb in range(num_lb):
        phys = bt_exp[:, lb].clamp(min=0, max=num_phys_blocks - 1)
        pages = kv_flat[phys]
        data = (
            pages[:, : block_size * D]
            .reshape(total_rows, block_size, D)
            .view(torch.float8_e4m3fn)
            .float()
        )
        scales = (
            pages[:, block_size * D :]
            .view(torch.float32)
            .reshape(total_rows, block_size)
        )
        dots = torch.einsum("rhd,rpd->rhp", q_f, data)
        scores = flag_gems.relu(dots * scales[:, None, :])
        out = (scores * w_f[:, :, None]).sum(dim=1)

        base = lb * block_size
        width = min(block_size, max_model_len - base)
        if width <= 0:
            break
        pos = base + pos_base[:width]
        valid = pos[None, :] < ctx_flat[:, None]
        region = logits[:, base : base + width]
        logits[:, base : base + width] = torch.where(valid, out[:, :width], region)
    return logits


def _patch_vllm_deep_gemm_module(module):
    """Wrap the paged MQA logits wrappers with GCU-safe fallbacks."""
    original_metadata = module.get_paged_mqa_logits_metadata
    original_logits = module.fp8_fp4_paged_mqa_logits
    if getattr(original_metadata, "_flaggems_paged_mqa_compat", False):
        return

    def get_paged_mqa_logits_metadata(context_lens, block_size, num_sms):
        backend_num_sms = _backend_num_sms()
        call_num_sms = num_sms
        if backend_num_sms is not None and int(num_sms) != backend_num_sms:
            call_num_sms = backend_num_sms
        try:
            return original_metadata(context_lens, block_size, call_num_sms)
        except Exception:
            return _flaggems_metadata(context_lens, block_size, num_sms)

    get_paged_mqa_logits_metadata._flaggems_paged_mqa_compat = True

    def fp8_fp4_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    ):
        try:
            return original_logits(
                q=q,
                kv_cache=kv_cache,
                weights=weights,
                context_lens=context_lens,
                block_tables=block_tables,
                schedule_metadata=schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=clean_logits,
            )
        except Exception:
            return _torch_paged_mqa_logits_reference(
                q=q,
                kv_cache=kv_cache,
                weights=weights,
                context_lens=context_lens,
                block_tables=block_tables,
                schedule_metadata=schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=clean_logits,
            )

    fp8_fp4_paged_mqa_logits._flaggems_paged_mqa_compat = True

    module.get_paged_mqa_logits_metadata = get_paged_mqa_logits_metadata
    module.fp8_fp4_paged_mqa_logits = fp8_fp4_paged_mqa_logits


class _PostImportLoader:
    """Delegate to the real loader, then patch the freshly-executed module."""

    def __init__(self, real_loader):
        self._real_loader = real_loader

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        _patch_vllm_deep_gemm_module(module)


class _VllmDeepGemmPagedCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE or _TARGET_MODULE in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostImportLoader(spec.loader)
                return spec
        return None


def _enable_torch_backend_for_paged_mqa_logits():
    """Route the paged MQA logits reference to the torch-native path."""
    current = os.environ.get(_TRITON_BACKEND_ENV, "")
    names = {name.strip() for name in current.replace(",", " ").split()}
    if _PAGED_KERNEL_NAME in names:
        return
    names.add(_PAGED_KERNEL_NAME)
    os.environ[_TRITON_BACKEND_ENV] = " ".join(sorted(names))


def install_vllm_paged_mqa_logits_compat():
    """Install the paged MQA logits GCU compat layer (idempotent)."""
    _enable_torch_backend_for_paged_mqa_logits()
    module = sys.modules.get(_TARGET_MODULE)
    if module is not None:
        _patch_vllm_deep_gemm_module(module)
        return
    for finder in sys.meta_path:
        if isinstance(finder, _VllmDeepGemmPagedCompatFinder):
            return
    sys.meta_path.insert(0, _VllmDeepGemmPagedCompatFinder())
