# Copyright 2026 FlagOS Contributors
#
# Kunlunxin (XPU) override of igammac (aten::special_gammaincc, out-of-place).
#
# Root causes of the generic `flag_gems/ops/igammac.py` on XPU:
#   1. `_build_d_coeffs` builds the DLMF 8.12.4 coefficient tensor with
#      `torch.tensor(list, dtype=..., device=flag_gems.device)`. Inside
#      `use_gems()`/`enable()` the Kunlunxin `to_copy` override refuses
#      CPU->XPU copies -> `NotImplementedError` before any kernel runs.
#   2. `tl_extra_shim.lgamma` resolves to `undefined symbol: __nv_lgammaf` at
#      xpu3 link time (same root cause as lgamma / special_gammainc /
#      special_gammaln / igammac_ overrides).
#   3. `tl_extra_shim.log1p` resolves to `undefined symbol: __nv_log1pf` at
#      link time; no `tl.log1p` exists in the XPU Triton dialect.
#   4. `@triton.autotune` with 10 configs re-compiles per shape on XPU and
#      inflates IR (harness lesson 2.1); replaced with a single BLOCK config.
#
# Fix: dedicated XPU kernel identical in math to the generic one (power
# series for x < a+1, Lentz continued fraction for x >= a+1, DLMF 8.12.4
# asymptotic expansion for a > 20 with |x-a|/a < 0.3, plus inf/nan/domain
# handling), with:
#   - inline Lanczos g=7 log-gamma for a > 0 (`_lgamma_pos`, same helper as
#     lgamma / special_gammainc / mvlgamma_ overrides),
#   - inline Taylor expansion of log(1+s)-s for the asymptotic eta (|s| fixed
#     < 0.32 in the activation region; 30 terms leaves < 1e-16 remainder),
#   - the coefficient table built eagerly at module import time (before any
#     override binding, so the cross-device to_copy override is not active),
#   - fixed BLOCK=512, no autotune; fp32 only (XPU has no fp64).
import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)

# DLMF 8.12.4 asymptotic expansion coefficients c_k(eta) = sum_n d[k,n] eta^n.
# Table 8.12.1 (25x25 row-major); only the first 8 rows are consumed by the
# kernel (ASYM_K=8), higher k contribute < 1e-13 in the activation region.
_D_TABLE = [
    [-0.3333333333333333, 0.08333333333333333, -0.014814814814814815, 0.0011574074074074073, 0.0003527336860670194, -0.0001787551440329218, 3.919263178522438e-05, -2.185448510679992e-06, -1.85406221071516e-06, 8.296711340953087e-07, -1.766595273682608e-07, 6.707853543401499e-09, 1.026180978424031e-08, -4.382036018453353e-09, 9.14769958223679e-10, -2.551419399494625e-11, -5.830772132550426e-11, 2.436194802066742e-11, -5.027669280114176e-12, 1.1004392031956135e-13, 3.371763262400985e-13, -1.392388722418162e-13, 2.853489380704744e-14, -5.139111834242573e-16, -1.975228829434944e-15,],
    [-0.001851851851851852, -0.003472222222222222, 0.0026455026455026454, -0.0009902263374485596, 0.00020576131687242798, -4.018775720164609e-07, -1.809855033448998e-05, 7.64916091608111e-06, -1.6120900894563446e-06, 4.647127802807434e-09, 1.378633446915721e-07, -5.752545603517705e-08, 1.195162859977815e-08, -1.754324171974765e-11, -1.009154371060041e-09, 4.162792991842583e-10, -8.56390702649298e-11, 6.06721510160476e-14, 7.162498964811485e-12, -2.933186643771437e-12, 5.996696365683689e-13, -2.167178652732331e-16, -4.978339972369262e-14, 2.029162882371342e-14, -4.13125571381061e-15,],
    [0.004133597883597884, -0.002681327160493827, 0.0007716049382716049, 2.0093878600823047e-06, -0.0001073665322636516, 5.292344882912012e-05, -1.276063518861873e-05, 3.423578734096138e-08, 1.372195730906293e-06, -6.298992138380055e-07, 1.428061420606424e-07, -2.047709842199087e-10, -1.409252991086752e-08, 6.228974084922022e-09, -1.367048839661711e-09, 9.42835615901468e-13, 1.287225240008932e-10, -5.564595613436332e-11, 1.197593554636698e-11, -4.168978225183864e-15, -1.094064042788459e-12, 4.662239946390136e-13, -9.905105763906907e-14, 1.893187676837352e-17, 8.859221872591127e-15,],
    [0.0006494341563786008, 0.00022947209362139917, -0.0004691894943952557, 0.00026772063206283885, -7.561801671883977e-05, -2.396505113867297e-07, 1.10826541153473e-05, -5.674952826991597e-06, 1.423090073243588e-06, -2.786108029152814e-11, -1.695840409193028e-07, 8.099464905388081e-08, -1.911116848597365e-08, 2.392862043980812e-12, 2.06201318154888e-09, -9.460496661855133e-10, 2.154104977577491e-10, -1.388823336813903e-14, -2.189476168196394e-11, 9.790998951171684e-12, -2.178219188018096e-12, 6.208819573407902e-17, 2.126978363279737e-13, -9.344688791517434e-14, 2.045367122678285e-14,],
    [-0.0008618882909167117, 0.0007840392217200666, -0.0002990724803031902, -1.463845257884342e-06, 6.641498215465122e-05, -3.968365047179435e-05, 1.137572697067842e-05, 2.507497226237533e-10, -1.695414953655831e-06, 8.90750753220531e-07, -2.292934834000805e-07, 2.956794137544049e-11, 2.886582974270878e-08, -1.418973943780322e-08, 3.44635804994649e-09, -2.302451717452807e-13, -3.940923302804641e-10, 1.86023389685045e-10, -4.356323005056618e-11, 1.278600101629623e-15, 4.67927502665792e-12, -2.149246470613483e-12, 4.908815614809652e-13, -6.33859148489156e-18, -5.045332069080094e-14,],
    [-0.00033679855336635813, -6.972813758365858e-05, 0.0002772753244959392, -0.00019932570516188847, 6.797780477937208e-05, 1.419062920643967e-07, -1.359404818976869e-05, 8.018470256334202e-06, -2.291481176508095e-06, -3.252473551298454e-10, 3.465284649108527e-07, -1.844718719117134e-07, 4.824096703789418e-08, -1.798946672174352e-14, -6.306194500013523e-09, 3.162417628774568e-09, -7.84092425369743e-10, 5.192679165254041e-15, 9.358944242306784e-11, -4.513426216163278e-11, 1.079912999311683e-11, -3.661886712685252e-17, -1.210902069055155e-12, 5.680743584990564e-13, -1.324965991634083e-13,],
    [0.0005313079364639922, -0.0005921664373536939, 0.0002708782096718045, 7.902353232660328e-07, -8.153969367561969e-05, 5.61168275310625e-05, -1.832911658284338e-05, -3.079613450603305e-09, 3.465155368803609e-06, -2.02913273960586e-06, 5.788792863149004e-07, 2.338630673826657e-13, -8.828600746330484e-08, 4.743595888040813e-08, -1.254541502071038e-08, 8.649648858010293e-14, 1.684605897926406e-09, -8.575492823577595e-10, 2.159822492923213e-10, -7.613230520476154e-16, -2.663982200853614e-11, 1.306570053661106e-11, -3.179916390236798e-12, 4.710976121367431e-18, 3.690280084276347e-13,],
    [0.00034436760689237765, 5.171790908260592e-05, -0.00033493161081142234, 0.0002812695154763237, -0.00010976582244684731, -1.274100909548549e-07, 2.774445151156364e-05, -1.826348880571133e-05, 5.787694949735052e-06, 4.93875893393627e-10, -1.059536701402604e-06, 6.166714376110408e-07, -1.756297335906046e-07, -1.297447328701544e-12, 2.695423606288966e-08, -1.457835290873127e-08, 3.887645959386175e-09, -3.881002251019412e-17, -5.327994173877287e-10, 2.743797764331484e-10, -6.995796092070568e-11, 2.589986387486848e-17, 8.856689099669637e-12, -4.403168815871311e-12, 1.086556194709165e-12,],
]

_D_COEFFS = None


def _get_d():
    """DLMF 8.12.4 coefficient table, built eagerly at import time.

    The table must be created before `flag_gems.enable()`/`use_gems()` binds
    the vendor `to_copy` override, otherwise the CPU->XPU copy of the source
    list raises NotImplementedError.
    """
    global _D_COEFFS
    if _D_COEFFS is None:
        flat = [float(v) for row in _D_TABLE for v in row]
        _D_COEFFS = torch.tensor(
            flat, dtype=torch.float32, device=flag_gems.device
        )
    return _D_COEFFS


# Eager creation at import time (no overrides active yet).
_D_COEFFS = _get_d()


@triton.jit
def _lgamma_pos(z):
    # Lanczos approximation of log-gamma for z > 0 (g=7, n=9 coefficients).
    # XPU Triton has no `lgamma` intrinsic (undefined symbol at link time),
    # so it is evaluated inline in fp32. gammaincc feeds a > 0 into
    # _lgamma_pos (the reflection branch is unnecessary since out-of-domain
    # a <= 0 produces NaN regardless).
    g = 7.0
    x = 0.99999999999980993
    x = x + 676.5203681218851 / (z + 0.0)
    x = x + (-1259.1392167224028) / (z + 1.0)
    x = x + 771.32342877765313 / (z + 2.0)
    x = x + (-176.61502916214059) / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x + (-0.13857109526572012) / (z + 5.0)
    x = x + 9.9843695780195716e-6 / (z + 6.0)
    x = x + 1.5056327351493116e-7 / (z + 7.0)
    t = (z - 1.0) + g + 0.5
    half_log_2pi = 0.9189385332046727
    return half_log_2pi + ((z - 1.0) + 0.5) * tl.log(t) - t + tl.log(x)


@triton.jit
def _log1p_minus_s(s):
    # log(1+s) - s = sum_{n>=2} (-1)^(n+1) s^n / n, by Taylor expansion.
    # Fixed in the asymptotic activation region (|s| <= 0.3); 30 terms leave
    # a remainder < 0.3^31/31 ~ 1e-17. Runtime range + sign-flip keeps the
    # loop compact (no unrolling, no per-iteration parity branch) and avoids
    # the fp32 1+s rounding and the absent log1p intrinsic on XPU.
    spow = s * s
    sgn = -1.0  # sign of the n=2 term (-s^2/2)
    acc = spow * sgn * 0.5
    for i in range(3, 32):
        i_f = tl.cast(i, tl.float32)
        spow = spow * s
        sgn = 0.0 - sgn
        acc = acc + sgn * spow / i_f
    return acc


@triton.jit
def igammac_kernel_xpu(
    a_ptr,
    x_ptr,
    out_ptr,
    d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    N_SER: tl.constexpr,
    N_CF: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # other=1.0 / other=0.0: masked lanes never feed valid results (their
    # stores are masked out as well).
    a = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    a_f = a.to(tl.float32)
    x_f = x.to(tl.float32)

    # Detect inf and NaN (same edge semantics as the generic kernel).
    is_nan_x = x_f != x_f
    is_nan_a = a_f != a_f
    is_inf_or_nan_x = (x_f * 0.0) != 0.0
    is_inf_or_nan_a = (a_f * 0.0) != 0.0
    is_inf_x = is_inf_or_nan_x & ~is_nan_x
    is_inf_a = is_inf_or_nan_a & ~is_nan_a
    is_finite = ~is_inf_or_nan_x & ~is_inf_or_nan_a
    in_domain = (a_f > 0.0) & (x_f >= 0.0) & is_finite

    log_gamma_a = _lgamma_pos(a_f)
    log_x_term = a_f * tl.log(x_f) - x_f - log_gamma_a

    # Path 1: power series for P(a, x) = e^{-x} x^a sum_n x^n / Gamma(a+n+1),
    # then Q = 1 - P. Converges for x < a + 1 (fixed-count loop; the terms
    # reach the fp32 rounding floor before N_SER, and the overflow regime
    # x >> a is masked out by use_series).
    term = 1.0 / a_f
    series_sum = term
    for i in range(1, N_SER):
        term = term * x_f / (a_f + tl.cast(i, tl.float32))
        series_sum = series_sum + term
    q_series = 1.0 - tl.exp(log_x_term) * series_sum
    q_series = tl.where(q_series > 1.0, 1.0, tl.where(q_series < 0.0, 0.0, q_series))

    # Path 2: Lentz's continued fraction for Q(a, x) directly. Converges
    # rapidly for x >= a + 1.
    tiny = 1e-30
    b0 = x_f + 1.0 - a_f
    f_val = b0
    C_val = b0
    D_val = tl.zeros_like(x_f)
    for i in range(1, N_CF):
        i_f = tl.cast(i, tl.float32)
        an = i_f * (a_f - i_f)
        bn = x_f + 2.0 * i_f + 1.0 - a_f

        D_val = bn + an * D_val
        D_val = tl.where(tl.abs(D_val) < tiny, tiny, D_val)
        C_val = bn + an / C_val
        C_val = tl.where(tl.abs(C_val) < tiny, tiny, C_val)

        D_val = 1.0 / D_val
        delta = C_val * D_val
        f_val = f_val * delta

    q_cf = tl.exp(log_x_term - tl.log(f_val))
    q_cf = tl.where(q_cf > 1.0, 1.0, tl.where(q_cf < 0.0, 0.0, q_cf))

    # Path 3: DLMF 8.12.4 asymptotic expansion for large a with x ~ a:
    #   Q(a,x) ~ 0.5 erfc(eta sqrt(a/2)) + e^{-a eta^2 / 2}
    #            * sum_{k=0}^{ASYM_K-1} c_k(eta) / a^k / sqrt(2 pi a),
    # with sigma = (x-a)/a, eta = sgn(x-a) sqrt(-2(log(1+sigma)-sigma)).
    sigma = (x_f - a_f) / a_f
    lam = x_f / a_f
    eta2 = -2.0 * _log1p_minus_s(sigma)
    eta = tl.where(
        lam > 1.0,
        tl.sqrt(eta2),
        tl.where(lam < 1.0, -tl.sqrt(eta2), 0.0),
    )

    q_asym = 0.5 * (1.0 - tl.math.erf(eta * tl.sqrt(a_f * 0.5)))
    poly_sum = 0.0
    afac = 1.0
    a_inv = 1.0 / a_f
    for k in tl.static_range(8):
        ck = 0.0
        eta_n = 1.0
        for n in tl.static_range(8):
            ck = ck + eta_n * tl.load(d_ptr + (k * 25 + n))
            eta_n = eta_n * eta
        poly_sum = poly_sum + ck * afac
        afac = afac * a_inv
    q_asym = q_asym + tl.exp(-0.5 * a_f * eta * eta) * poly_sum / tl.sqrt(
        2.0 * 3.141592653589793 * a_f
    )
    q_asym = tl.where(q_asym > 1.0, 1.0, tl.where(q_asym < 0.0, 0.0, q_asym))

    # Per-element path selection. Note: a plain two-level
    # `tl.where(use_asym, q_asym, tl.where(use_series, q_series, q_cf))`
    # blows the XPU uni_sram pass (three distinct live values); the
    # mask-and-sum form is NaN-safe (unselected lanes are forced to 0.0
    # before summing) and compiles fine.
    use_asym = (a_f > 20.0) & (tl.abs(x_f - a_f) / a_f < 0.3)
    use_asym = use_asym | ((a_f > 200.0) & (tl.abs(x_f - a_f) / a_f < 4.5 / tl.sqrt(a_f)))
    use_series = (x_f < (a_f + 1.0)) & ~use_asym
    use_cf = ~use_asym & ~use_series
    computed = (
        tl.where(use_asym, q_asym, 0.0)
        + tl.where(use_series, q_series, 0.0)
        + tl.where(use_cf, q_cf, 0.0)
    )

    # Boundary and infinity handling:
    # Q(a, 0) = 1, Q(inf, x) = 1, Q(a, inf) = 0, Q(inf, inf) = NaN,
    # out-of-domain (a <= 0 or x < 0) gives NaN.
    inf_result = tl.where(
        is_inf_x & is_inf_a,
        float("nan"),
        tl.where(is_inf_x, 0.0, tl.where(is_inf_a, 1.0, float("nan"))),
    )
    result = tl.where(
        is_finite,
        tl.where(in_domain, computed, float("nan")),
        inf_result,
    )

    tl.store(out_ptr + offsets, result.to(out_ptr.type.element_ty), mask=mask)


def _launch(out: torch.Tensor, a: torch.Tensor, x: torch.Tensor):
    a_c = a.contiguous()
    x_c = x.contiguous()
    was_noncontig = not out.is_contiguous()
    out_c = out.contiguous() if was_noncontig else out

    n = out_c.numel()
    if n > 0:
        BLOCK = 512
        grid = (triton.cdiv(n, BLOCK),)
        igammac_kernel_xpu[grid](
            a_c,
            x_c,
            out_c,
            _get_d(),
            n,
            BLOCK_SIZE=BLOCK,
            N_SER=50,
            N_CF=50,
            buffer_size_limit=2048,
        )

    if was_noncontig:
        out.copy_(out_c)
    return out


def igammac(a: torch.Tensor, x: torch.Tensor, *, out: torch.Tensor = None):
    logger.debug("GEMS_KUNLUNXIN IGAMMAC")
    if a.device.type != flag_gems.device:
        raise ValueError(f"igammac: first input tensor must be on {flag_gems.device}")
    if x.device.type != flag_gems.device:
        raise ValueError(f"igammac: second input tensor must be on {flag_gems.device}")

    if not a.is_floating_point():
        a = a.to(torch.get_default_dtype())
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    if a.dtype not in (torch.float32, torch.float64) or x.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise RuntimeError(
            f"igammac Triton kernel supports fp32/fp64, but got "
            f"a.dtype={a.dtype}, x.dtype={x.dtype}"
        )

    if out is None:
        out_dtype = torch.promote_types(a.dtype, x.dtype)
        out = torch.empty_like(a, dtype=out_dtype, device=a.device)
    else:
        if out.device.type != flag_gems.device:
            raise ValueError(
                f"igammac_out: output tensor must be on {flag_gems.device}"
            )
        if not out.is_floating_point():
            raise TypeError("igammac_out: output tensor must be a floating point type")
        if a.numel() != x.numel() or a.numel() != out.numel():
            raise ValueError(
                "igammac_out: input and output must have the same number of elements"
            )

    if a.dtype != out.dtype:
        a = a.to(out.dtype)
    if x.dtype != out.dtype:
        x = x.to(out.dtype)

    _launch(out, a, x)
    return out


def igammac_out(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN IGAMMAC_OUT")
    return igammac(a, x, out=out)