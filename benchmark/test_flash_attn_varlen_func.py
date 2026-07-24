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

"""Unified FlagGems/vLLM benchmark for variable-length FlashAttention."""

from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pytest
import torch

import flag_gems

from . import base, utils

SHAPE_ENTRY_KEY = "flash_attn_varlen_func"
SHAPE_FIELDS = (
    "name",
    "query_lens_rle",
    "kv_lens_rle",
    "num_query_heads",
    "num_kv_heads",
    "head_dim",
    "causal",
    "paged",
    "block_size",
    "num_blocks",
    "block_table_width",
    "kv_layout",
    "num_splits",
)
SHAPE_DESC = ", ".join(SHAPE_FIELDS)
TRACE_SHAPE_FIELDS = SHAPE_FIELDS + ("max_seqlen_q", "max_seqlen_k")
TRACE_SHAPE_DESC = ", ".join(TRACE_SHAPE_FIELDS)

vendor_name = flag_gems.vendor_name
HOPPER_AVAILABLE = (
    vendor_name == "nvidia"
    and flag_gems.device == "cuda"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
)

BENCHMARK_TARGETS = [
    pytest.param("flag_gems", 2, id="flag_gems-fa2"),
    pytest.param(
        "flag_gems",
        3,
        id="flag_gems-fa3",
        marks=pytest.mark.skipif(
            not HOPPER_AVAILABLE,
            reason="FA3 requires an NVIDIA Hopper GPU",
        ),
    ),
    pytest.param(
        "vllm",
        3,
        id="vllm-fa3",
        marks=pytest.mark.skipif(
            not HOPPER_AVAILABLE
            or utils.SkipVersion("vllm", "<0.9")
            or utils.SkipVersion("torch", "<2.7"),
            reason=(
                "vLLM FA3 requires vLLM >= 0.9, Torch >= 2.7, and an NVIDIA "
                "Hopper GPU"
            ),
        ),
    ),
    pytest.param(
        "iluvatar_legacy",
        2,
        id="iluvatar-legacy-fa2",
        marks=pytest.mark.skipif(
            vendor_name != "iluvatar",
            reason="Legacy FlashAttention baseline is only used on Iluvatar",
        ),
    ),
]


@dataclass(frozen=True)
class VarlenCase:
    name: str
    query_lens: Tuple[int, ...]
    kv_lens: Tuple[int, ...]
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool
    paged: bool
    block_size: Optional[int]
    num_blocks: Optional[int]
    block_table_width: Optional[int]
    kv_layout: str
    num_splits: int
    max_seqlen_q: Optional[int] = None
    max_seqlen_k: Optional[int] = None


@dataclass
class VarlenInput:
    case: VarlenCase
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: Optional[torch.Tensor]
    seqused_k: Optional[torch.Tensor]
    block_table: Optional[torch.Tensor]
    scheduler_metadata: Optional[torch.Tensor]
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True)
class Provider:
    name: str
    provider_version: str
    op: Callable
    scheduler_metadata_fn: Optional[Callable] = None


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if value < (0 if allow_zero else 1):
        kind = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {kind}, got {value!r}")
    return value


def _expand_rle(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty list of [length, repeat]")
    lengths = []
    for index, run in enumerate(value):
        if not isinstance(run, (list, tuple)) or len(run) != 2:
            raise ValueError(f"{field}[{index}] must be [length, repeat]")
        length = _positive_int(run[0], f"{field}[{index}].length")
        repeat = _positive_int(run[1], f"{field}[{index}].repeat")
        lengths.extend([length] * repeat)
    return tuple(lengths)


def _parse_case(
    record: Any,
    index: int,
    shape_fields: Tuple[str, ...] = SHAPE_FIELDS,
) -> VarlenCase:
    if shape_fields == SHAPE_FIELDS:
        expected_fields = SHAPE_FIELDS
    elif shape_fields == TRACE_SHAPE_FIELDS:
        expected_fields = TRACE_SHAPE_FIELDS
    else:
        raise ValueError(
            "varlen shape_desc must contain either the legacy fields "
            f"({SHAPE_DESC}) or trace fields ({TRACE_SHAPE_DESC})"
        )

    if not isinstance(record, (list, tuple)) or len(record) != len(expected_fields):
        raise ValueError(
            f"shape #{index} must contain {len(expected_fields)} fields: "
            f"{', '.join(expected_fields)}"
        )
    if expected_fields == TRACE_SHAPE_FIELDS:
        max_seqlen_q = record[-2]
        max_seqlen_k = record[-1]
        record = record[:-2]
    else:
        max_seqlen_q = None
        max_seqlen_k = None
    (
        name,
        query_lens_rle,
        kv_lens_rle,
        num_query_heads,
        num_kv_heads,
        head_dim,
        causal,
        paged,
        block_size,
        num_blocks,
        block_table_width,
        kv_layout,
        num_splits,
    ) = record

    if not isinstance(name, str) or not name:
        raise ValueError(f"shape #{index} name must be a non-empty string")

    query_lens = _expand_rle(query_lens_rle, f"shape {name!r}.query_lens_rle")
    kv_lens = _expand_rle(kv_lens_rle, f"shape {name!r}.kv_lens_rle")
    if len(query_lens) != len(kv_lens):
        raise ValueError(
            f"shape {name!r} has different query/KV batch sizes: "
            f"{len(query_lens)} != {len(kv_lens)}"
        )

    observed_max_seqlen_q = max(query_lens)
    observed_max_seqlen_k = max(kv_lens)
    if max_seqlen_q is not None:
        max_seqlen_q = _positive_int(
            max_seqlen_q, f"shape {name!r}.max_seqlen_q"
        )
        if max_seqlen_q < observed_max_seqlen_q:
            raise ValueError(
                f"shape {name!r} max_seqlen_q={max_seqlen_q} is smaller than "
                f"the observed maximum {observed_max_seqlen_q}"
            )
    if max_seqlen_k is not None:
        max_seqlen_k = _positive_int(
            max_seqlen_k, f"shape {name!r}.max_seqlen_k"
        )
        if max_seqlen_k < observed_max_seqlen_k:
            raise ValueError(
                f"shape {name!r} max_seqlen_k={max_seqlen_k} is smaller than "
                f"the observed maximum {observed_max_seqlen_k}"
            )

    num_query_heads = _positive_int(num_query_heads, f"shape {name!r}.num_query_heads")
    num_kv_heads = _positive_int(num_kv_heads, f"shape {name!r}.num_kv_heads")
    head_dim = _positive_int(head_dim, f"shape {name!r}.head_dim")
    if num_query_heads % num_kv_heads:
        raise ValueError(
            f"shape {name!r} requires num_query_heads to be divisible by "
            "num_kv_heads"
        )
    if not isinstance(causal, bool) or not isinstance(paged, bool):
        raise ValueError(f"shape {name!r} causal and paged must be booleans")
    if kv_layout not in ("NHD", "HND"):
        raise ValueError(f"shape {name!r} kv_layout must be NHD or HND")
    num_splits = _positive_int(
        num_splits, f"shape {name!r}.num_splits", allow_zero=True
    )

    if paged:
        block_size = _positive_int(block_size, f"shape {name!r}.block_size")
        num_blocks = _positive_int(num_blocks, f"shape {name!r}.num_blocks")
        block_table_width = _positive_int(
            block_table_width, f"shape {name!r}.block_table_width"
        )
        required_max_seqlen_k = (
            max_seqlen_k
            if max_seqlen_k is not None
            else observed_max_seqlen_k
        )
        required_width = max(
            (required_max_seqlen_k + block_size - 1) // block_size,
            max((length + block_size - 1) // block_size for length in kv_lens),
        )
        if block_table_width < required_width:
            raise ValueError(
                f"shape {name!r} needs block_table_width >= {required_width}"
            )
    elif any(
        value is not None for value in (block_size, num_blocks, block_table_width)
    ):
        raise ValueError(f"dense shape {name!r} must set paged cache fields to null")

    return VarlenCase(
        name=name,
        query_lens=query_lens,
        kv_lens=kv_lens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        paged=paged,
        block_size=block_size,
        num_blocks=num_blocks,
        block_table_width=block_table_width,
        kv_layout=kv_layout,
        num_splits=num_splits,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def _cu_seqlens(lengths: Tuple[int, ...], device: str) -> torch.Tensor:
    return torch.tensor((0, *accumulate(lengths)), dtype=torch.int32, device=device)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "source"


def flash_attn_varlen_legacy(
    flash_attn_varlen_func,
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    **_unused,
):
    """Adapt the legacy FlashAttention API used by the Iluvatar benchmark."""

    del q_v, return_softmax_lse
    k_flat = k.reshape(-1, k.shape[2], k.shape[3])
    v_flat = v.reshape(-1, v.shape[2], v.shape[3])
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
                torch.cumsum(seqused_k, dim=0),
            ]
        ).to(torch.int32)

    return flash_attn_varlen_func(
        q,
        k_flat,
        v_flat,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        tuple(window_size or (-1, -1)),
        float(softcap),
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        alibi_slopes is not None,
        0,
        1,
        out=out,
        bias=None,
    )


def _load_provider(name: str, fa_version: int) -> Provider:
    if name == "flag_gems":
        return Provider(
            name="flag_gems",
            provider_version=_package_version("flag_gems"),
            op=flag_gems.flash_attn_varlen_func,
        )
    elif name == "iluvatar_legacy":
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception as error:
            pytest.fail(f"legacy FlashAttention is unavailable: {error}")

        return Provider(
            name="vllm_legacy",
            provider_version=_package_version("flash-attn"),
            op=partial(flash_attn_varlen_legacy, flash_attn_varlen_func),
        )
    elif name == "vllm":
        try:
            from vllm.vllm_flash_attn import (
                fa_version_unsupported_reason,
                flash_attn_varlen_func,
                get_scheduler_metadata,
                is_fa_version_supported,
            )
        except Exception as error:
            pytest.fail(f"vLLM FlashAttention is unavailable: {error}")

        if not is_fa_version_supported(fa_version):
            reason = fa_version_unsupported_reason(fa_version) or "unknown reason"
            pytest.skip(f"vLLM FA{fa_version} is unavailable: {reason}")

        return Provider(
            name="vllm",
            provider_version=_package_version("vllm"),
            op=flash_attn_varlen_func,
            scheduler_metadata_fn=get_scheduler_metadata,
        )
    else:
        raise ValueError(f"unsupported FlashAttention provider: {name}")


class FlashAttnVarlenBenchmark(base.Benchmark):
    """Run every varlen shape through one selected public provider."""

    DEFAULT_METRICS = ["latency"]
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = SHAPE_DESC

    def __init__(
        self,
        *,
        provider: Provider,
        fa_version: int,
        cache_non_contiguous: bool = False,
        **kwargs,
    ):
        self.provider = provider
        self.fa_version = fa_version
        self.cache_non_contiguous = cache_non_contiguous
        super().__init__(
            torch_op=self._invoke,
            gems_op=self._invoke,
            dtypes=[torch.float16, torch.bfloat16],
            **kwargs,
        )

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        fields = tuple(field.strip() for field in self.shape_desc.split(","))
        if fields not in (SHAPE_FIELDS, TRACE_SHAPE_FIELDS):
            raise ValueError(
                "varlen shape_desc must contain either the legacy fields "
                f"({SHAPE_DESC}) or trace fields ({TRACE_SHAPE_DESC})"
            )
        cases = [
            _parse_case(record, index, fields)
            for index, record in enumerate(self.shapes)
        ]

        names = [case.name for case in cases]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"varlen shape names must be unique: {duplicates}")

        if self.fa_version == 2:
            cases = [case for case in cases if case.num_splits == 0]
            if not cases:
                pytest.skip("The shape file has no num_splits=0 FA2 cases.")
        self.shapes = cases

    def get_input_iter(self, dtype):
        for index, case in enumerate(self.shapes):
            yield (self._make_input(case, dtype, seed=2026 + index),)

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple[0]], {}

    def _make_input(self, case: VarlenCase, dtype, *, seed: int) -> VarlenInput:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        q = torch.randn(
            sum(case.query_lens),
            case.num_query_heads,
            case.head_dim,
            dtype=dtype,
            device=self.device,
            generator=generator,
        )
        out = torch.empty_like(q)
        cu_seqlens_q = _cu_seqlens(case.query_lens, self.device)
        max_seqlen_q = (
            case.max_seqlen_q
            if case.max_seqlen_q is not None
            else max(case.query_lens)
        )
        max_seqlen_k = (
            case.max_seqlen_k
            if case.max_seqlen_k is not None
            else max(case.kv_lens)
        )

        if case.paged:
            if case.kv_layout == "NHD":
                physical_shape = (
                    case.num_blocks,
                    case.block_size,
                    case.num_kv_heads,
                    case.head_dim,
                )
            else:
                physical_shape = (
                    case.num_blocks,
                    case.num_kv_heads,
                    case.block_size,
                    case.head_dim,
                )
            storage_shape = physical_shape
            if self.cache_non_contiguous:
                storage_shape = (case.num_blocks * 2, *physical_shape[1:])
            k = torch.randn(
                storage_shape,
                dtype=dtype,
                device=self.device,
                generator=generator,
            )
            v = torch.randn(
                storage_shape,
                dtype=dtype,
                device=self.device,
                generator=generator,
            )
            if self.cache_non_contiguous:
                k = k[::2]
                v = v[::2]
            if case.kv_layout == "HND":
                # Keep the public [page, token, head, dim] shape while using
                # the physical HND stride order selected by cache allocators.
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
            seqused_k = torch.tensor(
                case.kv_lens, dtype=torch.int32, device=self.device
            )
            cu_seqlens_k = None
            block_table = torch.randint(
                0,
                case.num_blocks,
                (len(case.kv_lens), case.block_table_width),
                dtype=torch.int32,
                device=self.device,
                generator=generator,
            )
        else:
            cache_shape = (
                sum(case.kv_lens),
                case.num_kv_heads,
                case.head_dim,
            )
            k = torch.randn(
                cache_shape,
                dtype=dtype,
                device=self.device,
                generator=generator,
            )
            v = torch.randn(
                cache_shape,
                dtype=dtype,
                device=self.device,
                generator=generator,
            )
            cu_seqlens_k = _cu_seqlens(case.kv_lens, self.device)
            seqused_k = None
            block_table = None

        scheduler_metadata = None
        if (
            self.provider.scheduler_metadata_fn is not None
            and self.fa_version == 3
            and case.paged
        ):
            scheduler_metadata = self.provider.scheduler_metadata_fn(
                batch_size=len(case.query_lens),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                num_heads_q=case.num_query_heads,
                num_heads_kv=case.num_kv_heads,
                headdim=case.head_dim,
                headdim_v=case.head_dim,
                qkv_dtype=dtype,
                cache_seqlens=seqused_k,
                cu_seqlens_q=cu_seqlens_q,
                page_size=case.block_size,
                causal=case.causal,
                window_size=(-1, -1),
                num_splits=case.num_splits,
            )

        return VarlenInput(
            case=case,
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            block_table=block_table,
            scheduler_metadata=scheduler_metadata,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    def _invoke(self, tensors: VarlenInput):
        case = tensors.case
        return self.provider.op(
            tensors.q,
            tensors.k,
            tensors.v,
            max_seqlen_q=tensors.max_seqlen_q,
            cu_seqlens_q=tensors.cu_seqlens_q,
            max_seqlen_k=tensors.max_seqlen_k,
            cu_seqlens_k=tensors.cu_seqlens_k,
            seqused_k=tensors.seqused_k,
            q_v=None,
            dropout_p=0.0,
            softmax_scale=case.head_dim**-0.5,
            causal=case.causal,
            window_size=[-1, -1],
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=tensors.block_table,
            return_softmax_lse=False,
            out=tensors.out,
            scheduler_metadata=tensors.scheduler_metadata,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            s_aux=None,
            num_splits=case.num_splits,
            cp_world_size=1,
            cp_rank=0,
            cp_tot_seqused_k=None,
            fa_version=self.fa_version,
        )

    def record_shapes(self, tensors: VarlenInput):
        case = tensors.case
        shape_detail = {
            "name": case.name,
            "provider": self.provider.name,
            "provider_version": self.provider.provider_version,
            "fa_version": self.fa_version,
            "num_splits": case.num_splits,
            "max_seqlen_q": tensors.max_seqlen_q,
            "max_seqlen_k": tensors.max_seqlen_k,
        }
        return shape_detail


@pytest.mark.skipif(vendor_name == "hygon", reason="#2816: RuntimeError")
@pytest.mark.skipif(vendor_name == "cambricon", reason="#2886: TypeError")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize(
    ("provider_name", "fa_version"),
    BENCHMARK_TARGETS,
)
def test_flash_attn_varlen_func(monkeypatch, provider_name, fa_version):
    """Benchmark one YAML as one pytest item through the public API."""

    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    provider = _load_provider(provider_name, fa_version)
    bench = FlashAttnVarlenBenchmark(
        op_name=SHAPE_ENTRY_KEY,
        provider=provider,
        fa_version=fa_version,
    )
    bench.run()


@pytest.mark.skipif(vendor_name == "hygon", reason="#2816: RuntimeError")
@pytest.mark.skipif(vendor_name == "cambricon", reason="#2886: TypeError")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.flash_attn_varlen_func_noncontig
@pytest.mark.parametrize(
    ("provider_name", "fa_version"),
    BENCHMARK_TARGETS,
)
def test_flash_attn_varlen_func_noncontig(monkeypatch, provider_name, fa_version):
    """Benchmark noncontiguous paged cache strides through the public API."""

    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    provider = _load_provider(provider_name, fa_version)
    bench = FlashAttnVarlenBenchmark(
        op_name="flash_attn_varlen_func_noncontig",
        provider=provider,
        fa_version=fa_version,
        cache_non_contiguous=True,
    )
    bench.run()
