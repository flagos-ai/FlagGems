# ZEROS_LIKE operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.zeros_like import (  # noqa: E402
    zeros_like as gems_zeros_like,
)
from flag_gems.experimental_ops.zeros_like import (  # noqa: E402
    zeros_like_out as gems_zeros_like_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (16, 8, 32, 32)])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "opts_case",
    ["none", "dtype_override", "contig_memfmt", "channels_last", "layout_device"],
)
def test_zeros_like_default_overload(shape, in_dtype, opts_case):
    inp = torch.randn(shape, dtype=in_dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    act_inp = inp.clone()

    def pick_alt_dtype(dt):
        if dt == torch.float32:
            return torch.float16
        if dt == torch.float16:
            return torch.float32
        return torch.float32

    kwargs = {}
    if opts_case == "none":
        kwargs = {}
    elif opts_case == "dtype_override":
        kwargs = {"dtype": pick_alt_dtype(in_dtype)}
    elif opts_case == "contig_memfmt":
        kwargs = {"device": "cuda", "memory_format": torch.contiguous_format}
    elif opts_case == "channels_last":
        if ref_inp.ndim == 4:
            kwargs = {"device": "cuda", "memory_format": torch.channels_last}
        else:
            kwargs = {"device": "cuda", "memory_format": torch.contiguous_format}
    elif opts_case == "layout_device":
        kwargs = {"layout": torch.strided, "device": "cuda"}

    # Align reference device with reference input when using CPU reference
    kwargs_ref = dict(kwargs)
    kwargs_ref["device"] = ref_inp.device

    ref_out = torch.ops.aten.zeros_like(ref_inp, **kwargs_ref)
    with flag_gems.use_gems():
        act_out = gems_zeros_like(act_inp, **kwargs)

    eff_dtype = kwargs.get("dtype", in_dtype)
    gems_assert_close(act_out, ref_out, dtype=eff_dtype)


@pytest.mark.zeros_like
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (16, 8, 32, 32)])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("memfmt_case", ["none", "contiguous", "channels_last"])
def test_zeros_like_out_overload(shape, in_dtype, out_dtype, memfmt_case):
    inp = torch.randn(shape, dtype=in_dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    act_inp = inp.clone()

    out_ref = torch.empty(shape, dtype=out_dtype, device=ref_inp.device)
    out_act = torch.empty(shape, dtype=out_dtype, device=flag_gems.device)

    if memfmt_case == "none":
        ref_out = torch.ops.aten.zeros_like.out(ref_inp, out=out_ref)
        with flag_gems.use_gems():
            act_out = gems_zeros_like_out(act_inp, out_act)
    else:
        memfmt = (
            torch.contiguous_format
            if memfmt_case == "contiguous"
            else (torch.channels_last if inp.ndim == 4 else torch.contiguous_format)
        )
        ref_out = torch.ops.aten.zeros_like.out(
            ref_inp, memory_format=memfmt, out=out_ref
        )
        with flag_gems.use_gems():
            act_out = gems_zeros_like_out(act_inp, out_act, memory_format=memfmt)

    gems_assert_close(act_out, ref_out, dtype=out_dtype)
