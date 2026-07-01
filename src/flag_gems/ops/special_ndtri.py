import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)


@triton.jit
def _special_ndtri_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y0 = x.to(tl.float32)

    # Cephes ndtri: inverse of standard normal CDF.
    # Three regions selected by the tail probability.
    EXP_M2 = 0.13533528323661269189  # exp(-2)
    S2PI = 2.50662827463100050242  # sqrt(2*pi)

    # Fold the upper tail onto the lower tail; `negate` undoes the fold.
    use_upper = y0 > (1.0 - EXP_M2)
    y = tl.where(use_upper, 1.0 - y0, y0)

    is_central = y > EXP_M2

    # ---------- central region ----------
    yc = y - 0.5
    y2 = yc * yc
    # P0(y2)
    p0 = -5.99633501014107895267e1
    p0 = p0 * y2 + 9.80010754185999661536e1
    p0 = p0 * y2 + -5.66762857469070293439e1
    p0 = p0 * y2 + 1.39312609387279679503e1
    p0 = p0 * y2 + -1.23916583867381258016e0
    # Q0(y2)
    q0 = 1.0
    q0 = q0 * y2 + 1.95448858338141759834e0
    q0 = q0 * y2 + 4.67627912898881538453e0
    q0 = q0 * y2 + 8.63602421390890590575e1
    q0 = q0 * y2 + -2.25462687854119370527e2
    q0 = q0 * y2 + 2.00260212380060660359e2
    q0 = q0 * y2 + -8.20372256168333339912e1
    q0 = q0 * y2 + 1.59056225126211695515e1
    q0 = q0 * y2 + -1.18331621121330003142e0
    central = (yc + yc * (y2 * p0 / q0)) * S2PI

    # ---------- tail region ----------
    # Guard the log against y <= 0 (handled later by edge masks).
    y_safe = tl.where(y > 0.0, y, 1.0)
    xt = tl.sqrt(-2.0 * tl.log(y_safe))
    x0 = xt - tl.log(xt) / xt
    z = 1.0 / xt
    near = xt < 8.0

    # P1/Q1 (z in tail-near)
    p1 = 4.05544892305962419923e0
    p1 = p1 * z + 3.15251094599893866154e1
    p1 = p1 * z + 5.71628192246421288162e1
    p1 = p1 * z + 4.40805073893200834700e1
    p1 = p1 * z + 1.46849561928858024014e1
    p1 = p1 * z + 2.18663306850790267539e0
    p1 = p1 * z + -1.40256079171354495875e-1
    p1 = p1 * z + -3.50424626827848203418e-2
    p1 = p1 * z + -8.57456785154685413611e-4
    q1 = 1.0
    q1 = q1 * z + 1.57799883256466749731e1
    q1 = q1 * z + 4.53907635128879210584e1
    q1 = q1 * z + 4.13172038254672030440e1
    q1 = q1 * z + 1.50425385692907503408e1
    q1 = q1 * z + 2.50464946208309415979e0
    q1 = q1 * z + -1.42182922854787788574e-1
    q1 = q1 * z + -3.80806407691578277194e-2
    q1 = q1 * z + -9.33259480895457427372e-4

    # P2/Q2 (z in tail-far)
    p2 = 3.23774891776946035970e0
    p2 = p2 * z + 6.91522889068984211695e0
    p2 = p2 * z + 3.93881025292474443415e0
    p2 = p2 * z + 1.33303460815807542389e0
    p2 = p2 * z + 2.01485389549179081538e-1
    p2 = p2 * z + 1.23716634817820021358e-2
    p2 = p2 * z + 3.01581553508235416007e-4
    p2 = p2 * z + 2.65806974686737550832e-6
    p2 = p2 * z + 6.23974539184983293730e-9
    q2 = 1.0
    q2 = q2 * z + 6.02427039364742014255e0
    q2 = q2 * z + 3.67983563856160859403e0
    q2 = q2 * z + 1.37702099489081330271e0
    q2 = q2 * z + 2.16236993594496635890e-1
    q2 = q2 * z + 1.34204006088543189037e-2
    q2 = q2 * z + 3.28014464682127739104e-4
    q2 = q2 * z + 2.89247864745380683936e-6
    q2 = q2 * z + 6.79019408009981274425e-9

    x1 = tl.where(near, z * p1 / q1, z * p2 / q2)
    tail = x0 - x1
    tail = tl.where(use_upper, tail, -tail)  # lower tail is negative

    y_out = tl.where(is_central, central, tail)

    # ---------- edge cases ----------
    nan = float("nan")
    pos_inf = float("inf")
    neg_inf = float("-inf")
    y_out = tl.where(y0 == 0.0, neg_inf, y_out)
    y_out = tl.where(y0 == 1.0, pos_inf, y_out)
    y_out = tl.where((y0 < 0.0) | (y0 > 1.0), nan, y_out)

    tl.store(out_ptr + offsets, y_out.to(x.dtype), mask=mask)


def _run_special_ndtri_kernel(x: torch.Tensor, out: torch.Tensor):
    if x.device.type != flag_gems.device or out.device.type != flag_gems.device:
        raise ValueError(f"Tensors must be {flag_gems.device} tensors")
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    x_c = x.contiguous()
    out_c = out.contiguous()
    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _special_ndtri_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_ndtri(x: torch.Tensor):
    """ATen wrapper: special_ndtri(Tensor self) -> Tensor"""
    logger.debug("GEMS SPECIAL_NDTRI")
    out = torch.empty_like(x)
    return _run_special_ndtri_kernel(x, out)


def special_ndtri_out(x: torch.Tensor, out: torch.Tensor):
    """ATen wrapper: special_ndtri.out(Tensor self, Tensor out) -> Tensor"""
    logger.debug("GEMS SPECIAL_NDTRI_OUT")
    if x.shape != out.shape:
        x = x.expand(out.shape)
    _run_special_ndtri_kernel(x, out)
    return out
