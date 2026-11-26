# SPDX-License-Identifier: Apache-2.0
"""Stage 1 BF16 histogram sampling kernels, with explicit invalid-row flags."""

import torch
import triton
import triton.language as tl

from flag_gems.fused.iluvatar_tile_sampler import _draw as _native_draw

BINS = 16384


@triton.jit
def _key(x):
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    high = (bits >> 16).to(tl.int32)
    mag = high & 32767
    negative = high > 32767
    bounded = (mag >= 0x3000) & (mag < 0x4800)
    is_zero = bits == 0
    minus_inf = high == 0xFF80
    valid = ((bits & 65535) == 0) & (bounded | is_zero | minus_inf)
    key = tl.where(negative, 6144 - (mag - 0x3000), 6146 + (mag - 0x3000))
    return tl.where(minus_inf, 0, tl.where(is_zero, 6145, key)), valid


@triton.jit
def _value(k):
    neg = k < 6145
    mag = tl.where(neg, 0x3000 + 6144 - k, 0x3000 + k - 6146)
    bits = tl.where(neg, mag | 32768, mag) << 16
    return tl.where(
        k == 0,
        -float("inf"),
        tl.where(k == 6145, 0.0, bits.to(tl.float32, bitcast=True)),
    )


@triton.jit
def _imax(a, b):
    return tl.maximum(a, b)


@triton.jit
def _count(
    X,
    COUNTS,
    BAD,
    stride,
    V: tl.constexpr,
    T: tl.constexpr,
    GROUP: tl.constexpr,
    ROW_FIRST: tl.constexpr,
):
    if ROW_FIRST:
        row, tile = tl.program_id(0), tl.program_id(1)
    else:
        tile, row = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, T)
    idx = tile * T + lane
    x = tl.load(X + row * stride + idx, idx < V, -float("inf"))
    key, valid = _key(x)
    bad = tl.sum((~valid & (idx < V)).to(tl.int32), 0)
    if bad:
        tl.atomic_or(BAD + row, 1, sem="relaxed")
    if GROUP == 2:
        # Keep a compact exact local histogram for the upper 1024 magnitude
        # codes of each sign. Rare lower-magnitude codes remain direct atomics.
        pos = valid & (idx < V) & (key >= 6146)
        neg = valid & (idx < V) & (key >= 1) & (key <= 6144)
        pbase = tl.max(tl.where(pos, key, 6146), 0) - 1023
        nbase = tl.min(tl.where(neg, key, 6144), 0)
        in_p = pos & (key >= pbase)
        in_n = neg & (key < nbase + 1024)
        local_key = tl.where(in_p, key - pbase, tl.where(in_n, 1024 + key - nbase, 0))
        b = tl.arange(0, 2048)
        hist = tl.histogram(local_key, 2048) - tl.where(
            b == 0, tl.sum((~(in_p | in_n)).to(tl.int32), 0), 0
        )
        dest = tl.where(b < 1024, pbase + b, nbase + b - 1024)
        tl.atomic_add(COUNTS + row * 16384 + dest, hist, hist > 0, sem="relaxed")
        tl.atomic_add(
            COUNTS + row * 16384 + key,
            1,
            valid & (idx < V) & ~in_p & ~in_n,
            sem="relaxed",
        )
    elif GROUP == 3:
        lanes = tl.arange(0, 16384)
        ok = valid & (idx < V)
        hist = tl.histogram(tl.where(ok, key, 0), 16384) - tl.where(
            lanes == 0, tl.sum((~ok).to(tl.int32), 0), 0
        )
        tl.atomic_add(COUNTS + row * 16384 + lanes, hist, hist > 0, sem="relaxed")
    elif GROUP == 1:
        keys = tl.sort(tl.where(valid & (idx < V), key, 16384), descending=False)
        prev = tl.gather(keys, tl.maximum(lane - 1, 0), 0)
        nxt = tl.gather(keys, tl.minimum(lane + 1, T - 1), 0)
        start = tl.associative_scan(
            tl.where((lane == 0) | (keys != prev), lane, 0), 0, _imax
        )
        active = ((lane == T - 1) | (nxt != keys)) & (keys < 16384)
        tl.atomic_add(
            COUNTS + row * 16384 + keys, lane - start + 1, active, sem="relaxed"
        )
    else:
        tl.atomic_add(COUNTS + row * 16384 + key, 1, valid & (idx < V), sem="relaxed")


@triton.jit
def _group_mass(COUNTS, PEAK, MASS, NBINS):
    group, row = tl.program_id(0), tl.program_id(1)
    key = group * 256 + tl.arange(0, 256)
    counts = tl.load(COUNTS + row * 16384 + key)
    live = (key > 0) & (counts > 0)
    last = tl.max(tl.where(live, key, 0), 0)
    highest = tl.where(last > 0, _value(last), 0.0)
    weights = tl.exp(tl.minimum(_value(key) - highest, 0.0))
    ix = row * 64 + group
    tl.store(PEAK + ix, last)
    tl.store(MASS + ix, tl.sum(tl.where(live, weights * counts.to(tl.float32), 0.0), 0))
    tl.store(NBINS + ix, tl.sum(live.to(tl.int32), 0))


@triton.jit
def _group_pivot(
    COUNTS,
    P,
    META,
    BAD,
    PEAK,
    MASS,
    NBINS,
    NATIVE_P=None,
    WRITE_P: tl.constexpr = False,
    TEMPERATURE=None,
    CHECK_T: tl.constexpr = False,
    UNIFORM=None,
    INTEGER_UNIFORM: tl.constexpr = False,
):
    row = tl.program_id(0)
    groups = tl.arange(0, 64)
    peaks = tl.load(PEAK + row * 64 + groups)
    highest = tl.max(peaks, 0)
    vmax = tl.where(highest > 0, _value(highest), 0.0)
    totals = tl.load(MASS + row * 64 + groups) * tl.exp(_value(peaks) - vmax)
    p = tl.load(P + row)
    # Kahan at the 64-group boundary. Within each group Triton uses tree reductions.
    z = 0.0
    err = 0.0
    for i in range(64):
        y = tl.sum(tl.where(groups == i, totals, 0.0), 0) - err
        temp = z + y
        err = (temp - z) - y
        z = temp
    target = (1.0 - p) * z
    c = tl.cumsum(totals, 0)
    gid = tl.minimum(tl.min(tl.where((c > target) & (totals > 0), groups, 64), 0), 63)
    before = tl.sum(tl.where(groups < gid, totals, 0.0), 0)
    key = gid * 256 + tl.arange(0, 256)
    counts = tl.load(COUNTS + row * 16384 + key)
    w = tl.where(
        (key > 0) & (counts > 0), tl.exp(tl.minimum(_value(key) - vmax, 0.0)), 0.0
    )
    prefix = before + tl.cumsum(w * counts.to(tl.float32), 0)
    pivot = tl.minimum(
        tl.min(tl.where((prefix > target) & (w > 0), key, 16384), 0), highest
    )
    weight = tl.sum(tl.where(key == pivot, w, 0.0), 0)
    subtotal = before + tl.sum(tl.where(key < pivot, w * counts.to(tl.float32), 0.0), 0)
    available = tl.sum(tl.where(key == pivot, counts, 0), 0)
    amount = (target - subtotal) / tl.maximum(weight, 1e-30)
    remove = tl.minimum(
        tl.maximum(tl.floor(amount).to(tl.int32), 0),
        available - (pivot == highest).to(tl.int32),
    )
    ambiguous = (tl.abs(amount - tl.floor(amount + 0.5)) * weight <= 1e-6 * z) & (
        p < 1.0
    )
    invalid = (
        (tl.sum(tl.load(NBINS + row * 64 + groups), 0) < 2)
        | (weight <= 0.0)
        | (p <= 0.0)
        | (p >= 1.0)
        | (p != p)
        | ambiguous
    )
    flag = tl.load(BAD + row) | invalid.to(tl.int32)
    if CHECK_T:
        flag = flag | (tl.load(TEMPERATURE + row) != 1.0).to(tl.int32)
    if INTEGER_UNIFORM:
        tl.store(UNIFORM + row, tl.load(UNIFORM + row) * (1.0 / 16777216.0))
    tl.store(BAD + row, flag)
    if WRITE_P:
        tl.store(NATIVE_P + row, tl.where(flag != 0, p, 1.0))
    tl.store(META + row * 3, pivot)
    tl.store(META + row * 3 + 1, remove)
    tl.store(META + row * 3 + 2, available)


def inputs(x):
    rows, v = x.shape
    tiles = triton.cdiv(v, 1024)
    return (
        torch.empty(rows, BINS, dtype=torch.int32, device=x.device),
        torch.empty(rows, dtype=torch.int32, device=x.device),
        torch.empty(rows, 3, dtype=torch.int32, device=x.device),
        torch.empty(rows, tiles, dtype=torch.int32, device=x.device),
        x.new_empty((rows, tiles)),
        x.new_empty((rows, tiles)),
        torch.empty(rows, dtype=torch.int64, device=x.device),
        torch.empty(rows, 64, dtype=torch.int32, device=x.device),
        x.new_empty((rows, 64)),
        torch.empty(rows, 64, dtype=torch.int32, device=x.device),
    )


def count(x, scratch, width=1024, group=0, row_first=True):
    counter, bad = scratch[:2]
    counter.zero_()
    bad.zero_()
    tiles = triton.cdiv(x.shape[1], width)
    grid = (x.shape[0], tiles) if row_first else (tiles, x.shape[0])
    _count[grid](
        x, counter, bad, x.stride(0), x.shape[1], width, group, row_first, num_warps=4
    )


@triton.jit
def _mass_for_tile(
    X, M, S, stride, V: tl.constexpr, T: tl.constexpr, WIDTH: tl.constexpr, tile, row
):
    cols = tile * WIDTH + tl.arange(0, WIDTH)
    x = tl.load(X + row.to(tl.int64) * stride + cols, cols < V, -float("inf"))
    maximum = tl.max(x, 0)
    invalid = tl.sum(((x != x) | (x == float("inf"))).to(tl.int32), 0) != 0
    finite_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    mass = tl.sum(tl.exp(x - finite_max), 0)
    tl.store(M + row * T + tile, tl.where(invalid, float("inf"), maximum))
    tl.store(S + row * T + tile, mass)


def _fill_uniforms(uniforms):
    if torch.cuda.is_current_stream_capturing():
        # The stock random_.from CUDA kernel advances the graph's device Philox offset on replay.
        # All 24-bit draws are exactly representable in float32.
        uniforms.random_(0, 1 << 24)
        return True
    uniforms.uniform_()
    return False


@triton.jit
def _mass_for_exceptional_rows(
    X, M, S, BAD, stride, V: tl.constexpr, NT: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(1)
    if tl.load(BAD + row):
        for tile in range(tl.program_id(0), NT, 8):
            _mass_for_tile(X, M, S, stride, V, NT, WIDTH, tile, row)


@triton.jit
def _draw_for_exceptional_rows(
    X,
    M,
    S,
    U,
    OUT,
    BAD,
    stride,
    V: tl.constexpr,
    NT: tl.constexpr,
    LANES: tl.constexpr,
    WIDTH: tl.constexpr,
):
    if tl.load(BAD + tl.program_id(0)):
        _native_draw(X, M, S, U, OUT, stride, V, NT, LANES, WIDTH)


def _draw_native_rows(filtered, peaks, mass, uniforms, out, bad):
    rows, vocab = filtered.shape
    tiles = triton.cdiv(vocab, 1024)
    _mass_for_exceptional_rows[(8, rows)](
        filtered, peaks, mass, bad, filtered.stride(0), vocab, tiles, 1024, num_warps=4
    )
    _draw_for_exceptional_rows[(rows,)](
        filtered,
        peaks,
        mass,
        uniforms,
        out,
        bad,
        filtered.stride(0),
        vocab,
        tiles,
        triton.next_power_of_2(tiles),
        1024,
        num_warps=4,
    )


@triton.jit
def _mass_and_ties(
    X, META, BAD, TIES, M, S, stride, V: tl.constexpr, T: tl.constexpr, NT: tl.constexpr
):
    tile, row = tl.program_id(0), tl.program_id(1)
    offset = row * NT + tile
    if tl.load(BAD + row):
        tl.store(TIES + offset, 0)
        return
    cols = tile * T + tl.arange(0, T)
    x = tl.load(X + row * stride + cols, cols < V, -float("inf")).to(tl.float32)
    pivot = _value(tl.load(META + row * 3))
    peak = tl.max(tl.where(x >= pivot, x, -float("inf")), 0)
    safe = tl.where(peak == -float("inf"), 0.0, peak)
    high = tl.sum(tl.where(x > pivot, tl.exp(x - safe), 0.0), 0)
    tied = tl.sum(((x == pivot) & (cols < V)).to(tl.int32), 0)
    tl.store(M + offset, peak)
    tl.store(S + offset, high)
    tl.store(TIES + offset, tied)


@triton.jit
def _select_tile_token(
    X,
    META,
    BAD,
    TIES,
    M,
    S,
    U,
    OUT,
    stride,
    V: tl.constexpr,
    T: tl.constexpr,
    NT: tl.constexpr,
    TT: tl.constexpr,
    FIRST_KEPT: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(BAD + row):
        return
    tiles = tl.arange(0, TT)
    maxima = tl.load(M + row * NT + tiles, tiles < NT, -float("inf"))
    mass = tl.load(S + row * NT + tiles, tiles < NT, 0.0)
    ties = tl.load(TIES + row * NT + tiles, tiles < NT, 0)
    before = tl.cumsum(ties) - ties
    pivot = _value(tl.load(META + row * 3))
    discard = tl.load(META + row * 3 + 1)
    total_ties = tl.load(META + row * 3 + 2)
    if FIRST_KEPT:
        kept = tl.minimum(tl.maximum(total_ties - discard - before, 0), ties)
    else:
        kept = ties - tl.minimum(tl.maximum(discard - before, 0), ties)
    peak = tl.max(maxima, 0)
    safe = tl.where(peak == -float("inf"), 0.0, peak)
    weights = mass * tl.exp(maxima - safe) + tl.where(
        kept > 0, kept.to(tl.float32) * tl.exp(pivot - safe), 0.0
    )
    prefix = tl.cumsum(weights)
    total = tl.max(prefix, 0)
    target = tl.load(U + row) * total
    first = tl.min(tl.where((weights > 0) & (prefix > target), tiles, NT), 0)
    selected = tl.where(first < NT, first, tl.max(tl.where(weights > 0, tiles, 0), 0))
    prior = tl.max(tl.where(tiles < selected, prefix, 0.0), 0)
    tile_weight = tl.sum(tl.where(tiles == selected, weights, 0.0), 0)
    ratio = (target - prior) / tl.where(tile_weight > 0.0, tile_weight, 1.0)
    cols = selected * T + tl.arange(0, T)
    x = tl.load(X + row * stride + cols, cols < V, -float("inf")).to(tl.float32)
    rank = tl.sum(tl.where(tiles == selected, before, 0), 0) + tl.cumsum(
        ((x == pivot) & (cols < V)).to(tl.int32)
    )
    if FIRST_KEPT:
        tie_keep = rank <= total_ties - discard
    else:
        tie_keep = rank > discard
    x = tl.where((x > pivot) | ((x == pivot) & tie_keep), x, -float("inf"))
    local_max = tl.max(x, 0)
    w = tl.exp(x - tl.where(local_max == -float("inf"), 0.0, local_max))
    cdf = tl.cumsum(w)
    threshold = tl.maximum(ratio, 0.0) * tl.max(cdf, 0)
    first_token = tl.min(tl.where((w > 0) & (cdf > threshold), cols, V), 0)
    last_token = tl.max(tl.where(w > 0, cols, 0), 0)
    tl.store(
        OUT + row,
        tl.where(
            (total > 0) & (peak < float("inf")),
            tl.where(first_token < V, first_token, last_token),
            0,
        ),
    )


def device_draw(
    x,
    p,
    u,
    scratch,
    native_filter,
    first_kept,
    width=512,
    fallback_p=None,
    integer_uniform=False,
):
    rows, v = x.shape
    counter, bad, meta, ties, peaks, high, out, group_peak, group_mass, group_number = (
        scratch
    )
    count(x, scratch, width, 0, True)
    _group_mass[(64, rows)](counter, group_peak, group_mass, group_number, num_warps=4)
    if fallback_p is None:
        fallback_p = torch.empty_like(p)
    _group_pivot[(rows,)](
        counter,
        p,
        meta,
        bad,
        group_peak,
        group_mass,
        group_number,
        fallback_p,
        True,
        None,
        False,
        u,
        integer_uniform,
        num_warps=4,
    )
    # The installed native Triton top-p kernel skips rows with p==1 on device.
    # It filters the exceptional rows in-place, before both sampling kernels.
    filtered = native_filter(x, None, fallback_p)
    _draw_native_rows(filtered, peaks, high, u, out, bad)
    nt = triton.cdiv(v, 1024)
    _mass_and_ties[(nt, rows)](
        filtered,
        meta,
        bad,
        ties,
        peaks,
        high,
        filtered.stride(0),
        v,
        1024,
        nt,
        num_warps=4,
    )
    _select_tile_token[(rows,)](
        filtered,
        meta,
        bad,
        ties,
        peaks,
        high,
        u,
        out,
        filtered.stride(0),
        v,
        1024,
        nt,
        triton.next_power_of_2(nt),
        first_kept,
        num_warps=4,
    )
    return out


_sampler_workspaces = {}


def _stream_key(device, vocab):
    import threading

    return (
        device.index,
        torch.cuda.current_stream(device).cuda_stream,
        threading.get_ident(),
        vocab,
    )


def reserve_sampler_workspace(device, rows=64, vocab=130560):
    key = _stream_key(device, vocab)
    if key not in _sampler_workspaces:
        exemplar = torch.empty((), device=device, dtype=torch.float32).expand(
            rows, vocab
        )
        _sampler_workspaces[key] = (
            inputs(exemplar),
            exemplar.new_empty((rows,)),
            exemplar.new_empty((rows,)),
        )


def sample_top_p(logits, p, native_filter, has_triton):
    rows, vocab = logits.shape
    if not (
        1 <= rows <= 64
        and vocab == 130560
        and p.dtype == torch.float32
        and p.is_contiguous()
        and p.shape == (rows,)
    ):
        return None
    key = _stream_key(logits.device, vocab)
    if key not in _sampler_workspaces:
        if torch.cuda.is_current_stream_capturing():
            return None
        reserve_sampler_workspace(logits.device, 64, vocab)
    scratch, uniform, p_native = _sampler_workspaces[key]
    active = [entry[:rows] for entry in scratch]
    # Returned token IDs have independent lifetimes even for adjacent calls on one stream.
    active[6] = torch.empty(64, device=logits.device, dtype=torch.int64)[:rows]
    uniforms = uniform[:rows]
    integer_uniform = _fill_uniforms(uniforms)
    return device_draw(
        logits,
        p,
        uniforms,
        active,
        native_filter,
        has_triton and rows >= 8,
        fallback_p=p_native[:rows],
        integer_uniform=integer_uniform,
    )


@triton.jit
def _copy_exceptional_logits(
    RAW, FLOAT, TEMP, BAD, stride, V: tl.constexpr, TILE: tl.constexpr
):
    row = tl.program_id(1)
    if tl.load(BAD + row):
        t = tl.load(TEMP + row)
        for tile in range(tl.program_id(0), triton.cdiv(V, TILE), 4):
            columns = tile * TILE + tl.arange(0, TILE)
            x = tl.load(RAW + row * stride + columns, columns < V, other=0.0).to(
                tl.float32
            )
            tl.store(FLOAT + row * V + columns, tl.div_rn(x, t), columns < V)


def draw_bf16(
    logits,
    p,
    temperature,
    uniforms,
    scratch,
    native_p,
    fp32,
    native_filter,
    output=None,
    integer_uniform=False,
):
    rows, vocab = logits.shape
    counter, bad, meta, ties, peaks, high, out, group_peak, group_mass, group_number = (
        scratch
    )
    if output is not None:
        out = output
    count(logits, scratch, 256, 0, True)
    _group_mass[(64, rows)](counter, group_peak, group_mass, group_number, num_warps=4)
    _group_pivot[(rows,)](
        counter,
        p,
        meta,
        bad,
        group_peak,
        group_mass,
        group_number,
        native_p,
        True,
        temperature,
        True,
        uniforms,
        integer_uniform,
        num_warps=4,
    )
    tiles = triton.cdiv(vocab, 1024)
    _copy_exceptional_logits[(4, rows)](
        logits, fp32, temperature, bad, logits.stride(0), vocab, 1024, num_warps=4
    )
    filtered = native_filter(fp32, None, native_p)
    _draw_native_rows(filtered, peaks, high, uniforms, out, bad)
    _mass_and_ties[(tiles, rows)](
        logits,
        meta,
        bad,
        ties,
        peaks,
        high,
        logits.stride(0),
        vocab,
        1024,
        tiles,
        num_warps=4,
    )
    _select_tile_token[(rows,)](
        logits,
        meta,
        bad,
        ties,
        peaks,
        high,
        uniforms,
        out,
        logits.stride(0),
        vocab,
        1024,
        tiles,
        triton.next_power_of_2(tiles),
        True,
        num_warps=4,
    )
    return out


_direct_views = {}


def prepare_bf16_sampling(device, vocab=130560):
    from flag_gems.fused.iluvatar_dense_prefill import borrow_prefill_float

    key = _stream_key(device, vocab)
    views = _direct_views.get(key)
    if views is not None or key not in _sampler_workspaces:
        return views
    staging = borrow_prefill_float(device, 64, vocab)
    if staging is None:
        return None
    arrays, uniforms, native_p = _sampler_workspaces[key]
    views = {64: (arrays, uniforms, native_p, staging)}
    for rows in range(8, 64):
        views[rows] = (
            tuple(entry[:rows] for entry in arrays),
            uniforms[:rows],
            native_p[:rows],
            staging[:rows],
        )
    _direct_views[key] = views
    return views


def sample_bf16(logits, p, temperature, native_filter):
    rows, vocab = logits.shape
    if not (
        logits.is_cuda
        and logits.dtype == torch.bfloat16
        and logits.stride(1) == 1
        and 8 <= rows <= 64
        and vocab == 130560
        and temperature is not None
        and p is not None
        and p.shape == temperature.shape == (rows,)
        and p.dtype == temperature.dtype == torch.float32
        and p.is_contiguous()
        and temperature.is_contiguous()
        and p.device == temperature.device == logits.device
    ):
        return None
    key = _stream_key(logits.device, vocab)
    views = _direct_views.get(key)
    if views is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        views = prepare_bf16_sampling(logits.device, vocab)
        if views is None:
            return None
    active, u, native_p, fp32 = views[rows]
    owned = torch.empty(64, dtype=torch.int32, device=logits.device)
    output = owned if rows == 64 else owned[:rows]
    integer_uniform = _fill_uniforms(u)
    return draw_bf16(
        logits,
        p,
        temperature,
        u,
        active,
        native_p,
        fp32,
        native_filter,
        output,
        integer_uniform,
    )
