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

import dataclasses
import importlib
import logging
import random

import pytest
import torch

import flag_gems
from flag_gems.utils.device_info import get_device_capability

from . import base

logger = logging.getLogger(__name__)


# NVIDIA gates FP8 E4M3 on sm_89+, and this check has historically been spelled
# `get_device_capability() >= (8, 9)`. That number only means "Ada or newer" on
# NVIDIA; other vendors report their own major/minor on a different scale, so
# applying the threshold to them is a false negative that skips this whole file
# on hardware that supports the dtype. Add a vendor here only after verifying
# on it that a Triton `tl.float8e4nv` conversion matches `torch.float8_e4m3fn`
# bit-for-bit -- MetaX C550 reports (8, 0) and does (verified 2026-08-05).
_FP8E4NV_CAPABLE_VENDORS = frozenset({"metax"})


def is_support_fp8e4nv():
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if flag_gems.vendor_name in _FP8E4NV_CAPABLE_VENDORS:
        return True
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


# The reference is registered under `torch.ops._C` by whichever compiled op
# library the platform ships, and probing that namespace does not import it --
# so where the reference IS installed, a bare hasattr() reports it missing and
# no comparison runs. The provider is not always vLLM's own: on MetaX it is
# `mcoplib._C`, while the `vllm` wheel there fails to load. Note importing the
# top-level package is not enough; the compiled submodule must be imported
# before the schemas register.
VENDOR_OP_LIBS = ("vllm._C", "mcoplib._C")


def _load_vendor_ref(op_name):
    """Return `torch.ops._C.<op_name>`, importing vendor libraries as needed.

    Returns None if no library provides it. Import failures are logged rather
    than swallowed -- a silent `except: pass` is what makes a missing baseline
    indistinguishable from an unimportable one.
    """
    fn = getattr(torch.ops._C, op_name, None)
    if callable(fn):
        return fn
    for lib in VENDOR_OP_LIBS:
        try:
            importlib.import_module(lib)
        except Exception as e:
            logger.info("vendor op library %s unavailable: %s", lib, e)
            continue
        fn = getattr(torch.ops._C, op_name, None)
        if callable(fn):
            logger.info("found %s in %s", op_name, lib)
            return fn
        logger.info("%s loaded but does not provide %s", lib, op_name)
    logger.info(
        "no vendor kernel for %s (tried %s)", op_name, ", ".join(VENDOR_OP_LIBS)
    )
    return None


_VENDOR_REF = _load_vendor_ref("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
VLLM_REF_AVAILABLE = _VENDOR_REF is not None
HEAD_DIM = 512
ROPE_DIM = 64
HEAD_BYTES = 584


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
    __test__ = False

    num_tokens: int
    num_heads: int
    num_tokens_insert: int
    block_size: int
    max_pos: int
    eps: float
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flag_gems.device


_random_counter = 0


class FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            _VENDOR_REF,
            [torch.bfloat16],
        )
        self.set_gems(flag_gems.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for (
            param
        ) in (
            FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.get_performance_test_params()
        ):
            yield from FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input(
                param
            )

    @staticmethod
    def get_performance_test_params():
        cases = [
            TestParam(
                num_tokens,
                num_heads,
                num_tokens_insert=num_tokens,
                block_size=64,
                max_pos=4096,
                eps=1e-6,
            )
            for num_tokens in [
                1,
                4,
                17,
                64,
                1024,
                2048,
                8192,
                32768,
                65536,
                98304,
                131072,
            ]
            for num_heads in [64, 128]
        ]
        return cases

    @staticmethod
    def init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
        if max_pos <= 8192:
            base = 10000.0
        elif max_pos <= 32768:
            base = 20000.0
        elif max_pos <= 65536:
            base = 40000.0
        elif max_pos <= 98304:
            base = 60000.0
        else:
            base = 100000.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
        return cache.to(dtype)

    @staticmethod
    def make_input(param: TestParam):
        num_tokens = param.num_tokens
        num_heads = param.num_heads
        num_tokens_insert = param.num_tokens_insert
        block_size = param.block_size
        max_pos = max(param.max_pos, num_tokens)
        eps = param.eps
        dtype = param.dtype
        device = param.device

        global _random_counter
        FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.init_seed(_random_counter)
        _random_counter = _random_counter + 1

        q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = (
            FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_cos_sin_cache(
                max_pos, ROPE_DIM, torch.float32, device
            )
        )

        num_blocks = (num_tokens + block_size - 1) // block_size + 1
        slot_mapping = torch.arange(num_tokens_insert, dtype=torch.int64, device=device)
        k_cache = torch.zeros(
            num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
        )
        yield (q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not VLLM_REF_AVAILABLE,
    reason="No vendor kernel found for this operator (tried %s)"
    % ", ".join(VENDOR_OP_LIBS),
)
@pytest.mark.skipif(not is_support_fp8e4nv(), reason="Device does not support fp8e4nv")
def test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert():
    bench = FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark()
    bench.run()
