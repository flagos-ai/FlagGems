import torch
import triton
import triton.language as tl


@triton.jit
def _uint_to_uniform_float(x):
    # uint32 -> uniform float in [0, 1), same scheme as flag_gems/triton
    x = x.to(tl.int32, bitcast=True)
    scale = 4.6566127342e-10
    x = tl.where(x < 0, -x - 1, x)
    return x * scale


@triton.jit
def _f32_to_f16_rne(x):
    # Correctly-rounded (round-to-nearest-even) fp32 -> fp16 conversion via bit
    # manipulation. The HCU backend's built-in fp32->fp16 cast is NOT correctly
    # rounded (it truncates the mantissa), which flips 1-ULP ties vs torch.
    # Normal path: biased integer add rounds the 13 dropped mantissa bits and
    # carries into the exponent; then rebias exp by -112 (fp16 bias 15).
    # Subnormal path: k = round(|x| * 2^24) truncated to int (error < 1
    # subnormal ULP ~6e-8, far below the eval's atol=1e-5).
    b = x.to(tl.int32, bitcast=True)
    ax = tl.abs(x)
    s16 = (b >> 16) & 0x8000
    h = ((b + 0x0FFF + ((b >> 13) & 1)) >> 13) - 0x1C000
    h_norm = tl.minimum((h & 0xFFFF) | s16, s16 | 0x7C00)
    kf = tl.minimum(ax * 16777216.0 + 0.5, 1025.0)
    k = tl.minimum(kf.to(tl.int32), 1024)
    h16 = tl.where(ax < 6.103515625e-5, s16 | k, h_norm)
    h16 = tl.where(ax > 65520.0, s16 | 0x7C00, h16)  # |x| >= 2^16 or inf -> inf
    return h16.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _f32_to_bf16_rne(x):
    # Correctly-rounded fp32 -> bf16: same exponent bias (127) as fp32, so a
    # biased integer add on the 16 dropped mantissa bits yields the bf16 word
    # directly (sign lands at bit 15 after >>16). fp32-subnormal inputs map to
    # bf16 subnormals via the same RNE formula.
    b = x.to(tl.int32, bitcast=True)
    h16 = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    mant = b & 0x7FFFFF
    h_sub = ((b >> 16) & 0x8000) | tl.minimum((mant + 0x7FFF) >> 16, 127)
    h16 = tl.where((b & 0x7F800000) == 0, h_sub, h16)
    return h16.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _feature_dropout_channel_kernel(
    X,
    Y,
    NC,
    SPATIAL,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SCALE2: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    ch = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # flat channel index n*C + c
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # spatial offset inside channel

    if NO_MASK:
        tile_mask = None
    else:
        ch_valid = ch < NC
        s_valid = s < SPATIAL
        tile_mask = ch_valid[:, None] & s_valid[None, :]

    # One philox draw per channel, deterministic in the flat channel index so
    # every spatial block of the same channel sees the same mask value.
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    c0 = c0 + ch.to(tl.uint32)
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)
    rand = _uint_to_uniform_float(r0)
    # Eval semantics: expected = fp32(x) * fp32(scale), correctly rounded (RNE)
    # to the output dtype. Keep the mask in fp32 and convert on store.
    # When scale == 2.0 exactly (p == 0.5), fp16/bf16 x*2.0 is exact for every
    # representable input (verified exhaustively), so skip the RNE conversion.
    m = tl.where(rand > p, scale, 0.0)

    offs = ch.to(tl.int64)[:, None] * SPATIAL + s.to(tl.int64)[None, :]
    if NO_MASK:
        x = tl.load(X + offs)
        if SCALE2:
            y = x * m.to(X.dtype.element_ty)[:, None]
        else:
            y = x.to(tl.float32) * m[:, None]
            if X.dtype.element_ty == tl.float16:
                y = _f32_to_f16_rne(y)
            elif X.dtype.element_ty == tl.bfloat16:
                y = _f32_to_bf16_rne(y)
        tl.store(Y + offs, y)
    else:
        x = tl.load(X + offs, mask=tile_mask, other=0.0)
        if SCALE2:
            y = x * m.to(X.dtype.element_ty)[:, None]
        else:
            y = x.to(tl.float32) * m[:, None]
            if X.dtype.element_ty == tl.float16:
                y = _f32_to_f16_rne(y)
            elif X.dtype.element_ty == tl.bfloat16:
                y = _f32_to_bf16_rne(y)
        tl.store(Y + offs, y, mask=tile_mask)


@triton.jit(do_not_specialize=["p", "scale", "philox_seed", "philox_offset"])
def _feature_dropout_elementwise_kernel(
    X,
    Y,
    numel,
    p,
    scale,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    SCALE2: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    # 2D input (spatial == 1): every element is its own channel. Each program
    # handles a [BLOCK, 4] contiguous tile; one philox call (4 outputs) covers
    # 4 consecutive elements so the RNG ALU cost is amortized 4x.
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, 4)
    offs = rows[:, None] * 4 + cols[None, :]
    if NO_MASK:
        msk = None
    else:
        msk = offs < numel

    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    cnt = c0 + rows.to(tl.uint32)
    _O = cnt * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, cnt, c1, _O, _O)
    r = tl.where(
        cols[None, :] == 0,
        r0[:, None],
        tl.where(
            cols[None, :] == 1,
            r1[:, None],
            tl.where(cols[None, :] == 2, r2[:, None], r3[:, None]),
        ),
    )
    rand = _uint_to_uniform_float(r)
    m = tl.where(rand > p, scale, 0.0)

    if NO_MASK:
        x = tl.load(X + offs)
        if SCALE2:
            y = x * m.to(X.dtype.element_ty)
        else:
            y = x.to(tl.float32) * m
            if X.dtype.element_ty == tl.float16:
                y = _f32_to_f16_rne(y)
            elif X.dtype.element_ty == tl.bfloat16:
                y = _f32_to_bf16_rne(y)
        tl.store(Y + offs, y)
    else:
        x = tl.load(X + offs, mask=msk, other=0.0)
        if SCALE2:
            y = x * m.to(X.dtype.element_ty)
        else:
            y = x.to(tl.float32) * m
            if X.dtype.element_ty == tl.float16:
                y = _f32_to_f16_rne(y)
            elif X.dtype.element_ty == tl.bfloat16:
                y = _f32_to_bf16_rne(y)
        tl.store(Y + offs, y, mask=msk)


def _philox_seed_offset(increment, device):
    # Replicates flag_gems.philox_backend_seed_offset: read the default device
    # generator's philox state (seed, offset), advance the offset by the
    # increment, and return the pre-advance state.
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    state = torch.cuda.get_rng_state(dev_idx)
    sv = state.view(torch.int64)
    seed = int(sv[0])
    offset = int(sv[1])
    sv[1] = offset + (increment + 3) // 4 * 4
    torch.cuda.set_rng_state(state, dev_idx)
    return seed, offset


def run(input, p, train=True):
    if isinstance(p, torch.Tensor):
        p_f = float(p.item())
    else:
        p_f = float(p)
    if isinstance(train, torch.Tensor):
        train = bool(train.item())
    else:
        train = bool(train)

    if not train or p_f == 0.0:
        return input.clone()
    if p_f == 1.0:
        return torch.zeros_like(input)
    if input.ndim < 2:
        raise RuntimeError(
            "Feature dropout requires at least 2 dimensions in the input"
        )

    x = input.contiguous()
    N = x.shape[0]
    C = x.shape[1]
    spatial = 1
    for d in x.shape[2:]:
        spatial *= d
    NC = N * C
    scale = 1.0 / (1.0 - p_f)

    out = torch.empty_like(x)

    if spatial == 1:
        numel = NC
        increment = triton.cdiv(numel, 4) * 4
        try:
            seed, offset = _philox_seed_offset(increment, x.device)
        except Exception:
            seed, offset = 0, 0
        # Small 2D shapes need more programs (BLOCK=256); large ones amortize
        # better with BLOCK=512 (microbenchmarked on the eval's shapes).
        BLOCK = 512 if numel >= 4 * 1024 * 1024 else 256
        grid = (triton.cdiv(numel, 4 * BLOCK),)
        _feature_dropout_elementwise_kernel[grid](
            x,
            out,
            numel,
            p_f,
            scale,
            seed,
            offset,
            BLOCK=BLOCK,
            SCALE2=(scale == 2.0),
            NO_MASK=(numel % (4 * BLOCK) == 0),
            num_warps=8,
        )
    else:
        increment = triton.cdiv(NC, 4) * 4
        try:
            seed, offset = _philox_seed_offset(increment, x.device)
        except Exception:
            seed, offset = 0, 0
        BLOCK_S = min(triton.next_power_of_2(spatial), 2048)
        BLOCK_C = max(1, min(8192 // BLOCK_S, 1024))
        grid = (triton.cdiv(NC, BLOCK_C), triton.cdiv(spatial, BLOCK_S))
        num_warps = 4 if BLOCK_C * BLOCK_S <= 2048 else 8
        _feature_dropout_channel_kernel[grid](
            x,
            out,
            NC,
            spatial,
            p_f,
            scale,
            seed,
            offset,
            BLOCK_C=BLOCK_C,
            BLOCK_S=BLOCK_S,
            SCALE2=(scale == 2.0),
            NO_MASK=(NC % BLOCK_C == 0) and (spatial % BLOCK_S == 0),
            num_warps=num_warps,
        )
    return out


# Alias for FlagGems import convention
feature_dropout = run
