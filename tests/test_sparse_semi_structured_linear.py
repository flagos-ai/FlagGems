import contextlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE, TO_CPU

# The reference is the native ``torch._sparse_semi_structured_linear`` (CUTLASS
# backend), which requires an NVIDIA SM8.x GPU. We probe availability by
# invoking the op on a small input that meets the CUTLASS backend's minimum
# shape constraint (fp16: N>=32, K>=64), and record the failure reason.
#
# When the native op is unavailable on this device, the whole module is skipped.
# When a given case (shape/dtype/activation) is outside what the native op
# supports, that case is skipped via ``_skip_if_native_cannot``.
NATIVE_AVAILABLE = False
_NATIVE_UNAVAILABLE_REASON = ""
if torch.cuda.is_available():
    try:
        with contextlib.ExitStack() as _stk:
            _stk.callback(
                setattr,
                torch.sparse.SparseSemiStructuredTensor,
                "_FORCE_CUTLASS",
                torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS,
            )
            torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
            _w = torch.randn(32, 64, dtype=torch.float16, device="cuda")
            _w[:, 2::4] = 0
            _w[:, 3::4] = 0
            _s = torch.sparse.to_sparse_semi_structured(_w)
            torch._sparse_semi_structured_linear(
                torch.randn(8, 64, dtype=torch.float16, device="cuda"),
                _s.packed,
                _s.meta,
            )
        NATIVE_AVAILABLE = True
    except Exception as _e:
        NATIVE_AVAILABLE = False
        _NATIVE_UNAVAILABLE_REASON = f"{type(_e).__name__}: {_e}"

if not NATIVE_AVAILABLE:
    pytestmark = pytest.mark.skip(
        reason=(
            "torch._sparse_semi_structured_linear (native 2:4 reference) is not "
            "available on this device/build; requires NVIDIA SM8.x. "
            f"({_NATIVE_UNAVAILABLE_REASON})"
        )
    )

# The native op only supports fp16 / bf16 on CUDA.
FLOAT_DTYPES = [torch.float16, torch.bfloat16]

if QUICK_MODE:
    SPARSE_LINEAR_SHAPES = [(16, 32)]
else:
    # K must be a multiple of 4. Representative shapes from small to medium.
    SPARSE_LINEAR_SHAPES = [
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
    ]


def _build_2_4_weight(N, K, dtype, device, *, choice=None):
    """Build a dense weight with a valid 2:4 pattern and its bool meta.

    Each group of 4 consecutive K elements keeps exactly 2. ``choice[n, k]`` True
    keeps positions 0,1; False keeps positions 2,3. Different rows use
    independent patterns, exercising per-row 2:4 selection.

    Returns (dense_weight (N,K), choice_bool (N, K//4)).
    """
    K4 = K // 4
    base = torch.randn(N, K, dtype=dtype, device=device)
    if choice is None:
        choice = torch.randint(0, 2, (N, K4), device=device).bool()
    w = torch.zeros(N, K, dtype=dtype, device=device)
    for g in range(K4):
        keep = choice[:, g]
        w[:, 4 * g] = torch.where(
            keep, base[:, 4 * g], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 1] = torch.where(
            keep, base[:, 4 * g + 1], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 2] = torch.where(
            ~keep, base[:, 4 * g + 2], torch.zeros_like(keep, dtype=dtype)
        )
        w[:, 4 * g + 3] = torch.where(
            ~keep, base[:, 4 * g + 3], torch.zeros_like(keep, dtype=dtype)
        )
    return w, choice


def _pytorch_ref(input, weight, choice, bias=None, activation=None, out_dtype=None):
    """Pure-PyTorch 2:4 reference.

    Folds the select-position mask into the weight and applies the
    post-processing (bias / activation / out_dtype) with plain PyTorch in
    float32.
    """
    N, K = weight.shape
    K4 = K // 4
    wr = weight.view(N, K4, 4)
    keep = torch.cat(
        [
            choice.unsqueeze(2),
            choice.unsqueeze(2),
            (~choice).unsqueeze(2),
            (~choice).unsqueeze(2),
        ],
        dim=2,
    )
    masked = torch.where(keep, wr, torch.zeros_like(wr)).reshape(N, K)
    out = input.float() @ masked.t().float()
    if bias is not None:
        out = out + bias.float()
    out = out.to(out_dtype if out_dtype is not None else input.dtype)
    if activation == "relu":
        out = torch.relu(out)
    elif activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)
    elif activation == "gelu":
        out = torch.nn.functional.gelu(out)
    return out


# CUTLASS backend minimum sparse shape for fp16/bf16: N >= 32, K >= 64 (and
# multiples thereof). Shapes below this cannot run on the native op; such cases
# are skipped via ``_skip_if_native_cannot``.
_NATIVE_MIN_ROWS = 32
_NATIVE_MIN_COLS = 64


def _native_ref(input, weight, choice, bias=None, activation=None):
    """Reference via the native op, forcing the CUTLASS backend.

    Sets ``_FORCE_CUTLASS`` for this call and restores it afterwards.
    ``activation`` (when given) is forwarded to the native op, which fuses it
    into the sparse matmul.
    """
    saved = torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS
    torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
    try:
        s = torch.sparse.to_sparse_semi_structured(weight)
        out = torch._sparse_semi_structured_linear(
            input, s.packed, s.meta, bias=bias, activation=activation
        )
        return out
    finally:
        torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = saved


def _ref(input, weight, choice, bias=None, activation=None):
    """Reference via the native ``torch._sparse_semi_structured_linear`` (CUTLASS
    backend). Callers gate the call with ``_skip_if_native_cannot`` so that any
    shape/dtype/activation the native op cannot serve is skipped first.
    """
    return _native_ref(input, weight, choice, bias=bias, activation=activation)


def _skip_if_native_cannot(weight, *, dtype=None, activation=None, out_dtype=None):
    """Skip the current case when the native op cannot serve it.

    The native ``torch._sparse_semi_structured_linear`` (CUTLASS backend) only
    supports fp16/bf16, shapes with N>=32 and K>=64 (each a multiple thereof),
    the fused activations {None, 'relu', 'silu'} (NOT 'gelu'), and an int8->int32
    quantized out_dtype path. Anything else is skipped here.
    """
    if not NATIVE_AVAILABLE:
        pytest.skip("native _sparse_semi_structured_linear unavailable on this device")
    n, k = weight.shape
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
        pytest.skip(f"native op does not support dtype {dtype}")
    if not (
        n >= _NATIVE_MIN_ROWS
        and k >= _NATIVE_MIN_COLS
        and n % _NATIVE_MIN_ROWS == 0
        and k % _NATIVE_MIN_COLS == 0
    ):
        pytest.skip(
            f"native op shape constraint unmet for weight {tuple(weight.shape)} "
            f"(needs N>={_NATIVE_MIN_ROWS}/K>={_NATIVE_MIN_COLS}, each a multiple)"
        )
    if activation is not None and activation not in ("relu", "silu", "swish", "gelu"):
        pytest.skip(f"native op has no counterpart for activation '{activation}'")
    if out_dtype is not None:
        pytest.skip(
            "native op only supports int8->int32 out_dtype, not float out_dtype"
        )


def _assert_close(res, ref, dtype, atol):
    """Compare gems result against the reference.

    The reference is computed on CUDA (the native op has no CPU kernel). Under
    ``--ref=cpu`` we move both tensors to CPU first, matching the convention in
    ``test_act_quant`` so ``gems_assert_close``'s ``to_cpu`` sees a CPU reference.
    """
    if TO_CPU:
        ref = ref.to("cpu")
        res = res.to("cpu")
    utils.gems_assert_close(res, ref, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("M, K", SPARSE_LINEAR_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sparse_semi_structured_linear(M, K, dtype):
    """Compare the gems kernel against the native 2:4 op with per-row patterns."""
    N = K

    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight, choice = _build_2_4_weight(N, K, dtype, flag_gems.device)
    meta = choice.to(torch.int8)

    # Reference is the native aten op; skip shapes the native op cannot serve.
    _skip_if_native_cannot(weight, dtype=dtype)
    # Reference stays on the same CUDA device: native op has no CPU kernel.
    ref_out = _ref(input, weight, choice)

    res_out = flag_gems._sparse_semi_structured_linear(input, weight, meta)

    # Both tensors live on CUDA here; _assert_close moves them to CPU under
    # --ref=cpu. The 2:4 element-wise select-position accumulation differs from
    # the native op's tensor-core path in rounding, so use a relaxed tolerance.
    atol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 0.02
    _assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("M, K", [(32, 64)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sparse_semi_structured_linear_with_bias(M, K, dtype):
    """Test sparse semi-structured linear with bias against the native op."""
    N = K

    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight, choice = _build_2_4_weight(N, K, dtype, flag_gems.device)
    bias = torch.randn(N, dtype=dtype, device=flag_gems.device)
    meta = choice.to(torch.int8)

    _skip_if_native_cannot(weight, dtype=dtype)
    ref_out = _ref(input, weight, choice, bias=bias)

    res_out = flag_gems._sparse_semi_structured_linear(input, weight, meta, bias=bias)

    atol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 0.02
    _assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("M, K, N", [(16, 64, 48), (32, 128, 96)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sparse_semi_structured_linear_n_ne_k(M, K, N, dtype):
    """N and K are independent dimensions; exercise N != K."""
    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight, choice = _build_2_4_weight(N, K, dtype, flag_gems.device)
    meta = choice.to(torch.int8)

    _skip_if_native_cannot(weight, dtype=dtype)
    ref_out = _ref(input, weight, choice)

    res_out = flag_gems._sparse_semi_structured_linear(input, weight, meta)

    atol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 0.02
    _assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sparse_semi_structured_linear_activation(activation, dtype):
    """The activation post-processing folds onto the sparse matmul output.

    ``relu`` and ``silu`` are fused by the native op, so the activation is
    forwarded to it. ``gelu`` is not part of the native fused set (the native op
    only fuses {None, relu, silu}), so the reference runs the native op without
    activation and applies ``torch.nn.functional.gelu`` on the matmul output.
    """
    M, K, N = 16, 64, 64
    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight, choice = _build_2_4_weight(N, K, dtype, flag_gems.device)
    meta = choice.to(torch.int8)

    _skip_if_native_cannot(weight, dtype=dtype, activation=activation)
    if activation == "gelu":
        # native op has no 'gelu' fused path: do the sparse matmul natively
        # (activation=None, which the native op supports), then apply gelu via
        # the standard aten op on the matmul output.
        ref_out = _ref(input, weight, choice)  # activation=None
        ref_out = torch.nn.functional.gelu(ref_out)
    else:
        # relu / silu are fused by the native op; forward the activation.
        ref_out = _ref(input, weight, choice, activation=activation)

    res_out = flag_gems._sparse_semi_structured_linear(
        input, weight, meta, activation=activation
    )

    atol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 0.02
    _assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("M, K", [(64, 128), (128, 256)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sparse_semi_structured_linear_with_bias_shapes(M, K, dtype):
    """Bias path across larger shapes (N == K here)."""
    N = K
    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)
    weight, choice = _build_2_4_weight(N, K, dtype, flag_gems.device)
    bias = torch.randn(N, dtype=dtype, device=flag_gems.device)
    meta = choice.to(torch.int8)

    _skip_if_native_cannot(weight, dtype=dtype)
    ref_out = _ref(input, weight, choice, bias=bias)

    res_out = flag_gems._sparse_semi_structured_linear(input, weight, meta, bias=bias)

    atol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 0.02
    _assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_linear
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_semi_structured_linear_non_contiguous_weight(dtype):
    """The kernel indexes weight by explicit strides, so a non-contiguous
    (transposed) weight must produce the same result as its contiguous copy."""
    M, K, N = 16, 64, 64
    torch.manual_seed(12345)
    input = torch.randn(M, K, dtype=dtype, device=flag_gems.device)

    # Build the 2:4 weight as a contiguous (N, K) tensor, then derive a
    # non-contiguous (N, K) view by transposing an (K, N) base and re-applying
    # the pattern so both carry identical values.
    w_contig, choice = _build_2_4_weight(N, K, dtype, flag_gems.device)
    base = w_contig.t().contiguous()
    w_nc = base.t()
    for g in range(K // 4):
        keep = choice[:, g]
        z = torch.zeros(N, dtype=dtype, device=flag_gems.device)
        w_nc[:, 4 * g] = torch.where(keep, w_nc[:, 4 * g], z)
        w_nc[:, 4 * g + 1] = torch.where(keep, w_nc[:, 4 * g + 1], z)
        w_nc[:, 4 * g + 2] = torch.where(~keep, w_nc[:, 4 * g + 2], z)
        w_nc[:, 4 * g + 3] = torch.where(~keep, w_nc[:, 4 * g + 3], z)
    assert not w_nc.is_contiguous()
    meta = choice.to(torch.int8)

    _skip_if_native_cannot(w_contig, dtype=dtype)
    ref_out = _ref(input, w_contig, choice)

    res_nc = flag_gems._sparse_semi_structured_linear(input, w_nc, meta)

    atol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 0.02
    _assert_close(res_nc, ref_out, dtype, atol=atol)
