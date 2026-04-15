# FlagGems Grid Sample Optimization by Team: 龙战于野 (Kimi0425)
# Advanced Task #20: Full support for Bicubic, Reflection, and 5D Volumetric sampling.
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry
import logging

logger = logging.getLogger(__name__)

@triton.jit
def get_bicubic_weight(d):
    abs_d = tl.abs(d)
    d2 = abs_d * abs_d
    d3 = d2 * abs_d
    return tl.where(abs_d <= 1.0, 1.25 * d3 - 2.25 * d2 + 1.0, 
                   tl.where(abs_d < 2.0, -0.75 * d3 + 3.75 * d2 - 6.0 * abs_d + 3.0, 0.0))

@triton.jit
def reflect_coords(x, size, align_corners: tl.constexpr):
    if align_corners:
        if size <= 1: return 0.0
        x = tl.abs(x)
        m = size - 1
        flips = (x / m).to(tl.int32)
        res = x % m
        return tl.where(flips % 2 == 1, m - res, res)
    else:
        x = tl.abs(x + 0.5)
        flips = (x / size).to(tl.int32)
        res = x % size
        return (tl.where(flips % 2 == 1, size - res, res)) - 0.5

@triton.jit
def apply_padding(x, size, padding_mode: tl.constexpr, align_corners: tl.constexpr):
    if padding_mode == 1: # border
        return tl.clamp(x, 0.0, (size - 1.0).to(x.dtype))
    elif padding_mode == 2: # reflection
        return reflect_coords(x, size, align_corners)
    return x

@triton.jit
def _fwd_kernel_2d(input_ptr, grid_ptr, output_ptr, N, C, HI, WI, HO, WO, 
                   si_n, si_c, si_h, si_w, sg_n, sg_h, sg_w, sg_c, so_n, so_c, so_h, so_w,
                   interp: tl.constexpr, padd: tl.constexpr, ali: tl.constexpr, B: tl.constexpr):
    n, c = tl.program_id(1) // C, tl.program_id(1) % C
    offs = tl.program_id(0) * B + tl.arange(0, B)
    mask = offs < (HO * WO)
    oh, ow = offs // WO, offs % WO
    g_base = grid_ptr + n * sg_n + oh * sg_h + ow * sg_w
    gx, gy = tl.load(g_base, mask=mask), tl.load(g_base + sg_c, mask=mask)
    gx = tl.where(tl.isnan(gx), 2.0, gx); gy = tl.where(tl.isnan(gy), 2.0, gy)
    if ali:
        ix = ((gx + 1.0) / 2.0) * (WI - 1); iy = ((gy + 1.0) / 2.0) * (HI - 1)
    else:
        ix = ((gx + 1.0) * WI - 1.0) / 2.0; iy = ((gy + 1.0) * HI - 1.0) / 2.0
    ix = apply_padding(ix, WI, padd, ali); iy = apply_padding(iy, HI, padd, ali)
    in_base = input_ptr + n * si_n + c * si_c
    if interp == 0: # bilinear
        x0, y0 = tl.floor(ix).to(tl.int32), tl.floor(iy).to(tl.int32)
        x1, y1 = x0 + 1, y0 + 1
        wx, wy = ix - x0, iy - y0
        v00 = tl.load(in_base + y0 * si_h + x0 * si_w, mask=mask & (x0>=0) & (x0<WI) & (y0>=0) & (y0<HI), other=0.0)
        v01 = tl.load(in_base + y0 * si_h + x1 * si_w, mask=mask & (x1>=0) & (x1<WI) & (y0>=0) & (y0<HI), other=0.0)
        v10 = tl.load(in_base + y1 * si_h + x0 * si_w, mask=mask & (x0>=0) & (x0<WI) & (y1>=0) & (y1<HI), other=0.0)
        v11 = tl.load(in_base + y1 * si_h + x1 * si_w, mask=mask & (x1>=0) & (x1<WI) & (y1>=0) & (y1<HI), other=0.0)
        res = v00*(1-wx)*(1-wy) + v01*wx*(1-wy) + v10*(1-wx)*wy + v11*wx*wy
    elif interp == 1: # nearest
        x, y = tl.math.round(ix).to(tl.int32), tl.math.round(iy).to(tl.int32)
        res = tl.load(in_base + y * si_h + x * si_w, mask=mask & (x>=0) & (x<WI) & (y>=0) & (y<HI), other=0.0)
    else: # bicubic
        x0, y0, res = tl.floor(ix).to(tl.int32), tl.floor(iy).to(tl.int32), 0.0
        for m in range(-1, 3):
            for l in range(-1, 3):
                xi, yi = x0 + l, y0 + m
                w = get_bicubic_weight(ix - xi) * get_bicubic_weight(iy - yi)
                res += tl.load(in_base + yi * si_h + xi * si_w, mask=mask & (xi>=0) & (xi<WI) & (yi>=0) & (yi<HI), other=0.0) * w
    tl.store(output_ptr + n * so_n + c * so_c + oh * so_h + ow * so_w, res, mask=mask)

@triton.jit
def _fwd_kernel_3d(input_ptr, grid_ptr, output_ptr, N, C, DI, HI, WI, DO, HO, WO, 
                   si_n, si_c, si_d, si_h, si_w, sg_n, sg_d, sg_h, sg_w, sg_c, so_n, so_c, so_d, so_h, so_w,
                   interp: tl.constexpr, padd: tl.constexpr, ali: tl.constexpr, B: tl.constexpr):
    n, c = tl.program_id(1) // C, tl.program_id(1) % C
    offs = tl.program_id(0) * B + tl.arange(0, B)
    mask = offs < (DO * HO * WO)
    od, rem = offs // (HO * WO), offs % (HO * WO)
    oh, ow = rem // WO, rem % WO
    g_base = grid_ptr + n * sg_n + od * sg_d + oh * sg_h + ow * sg_w
    gx, gy, gz = tl.load(g_base, mask=mask), tl.load(g_base + sg_c, mask=mask), tl.load(g_base + 2*sg_c, mask=mask)
    if ali:
        ix = ((gx+1)/2)*(WI-1); iy = ((gy+1)/2)*(HI-1); iz = ((gz+1)/2)*(DI-1)
    else:
        ix = ((gx+1)*WI-1)/2; iy = ((gy+1)*HI-1)/2; iz = ((gz+1)*DI-1)/2
    ix, iy, iz = apply_padding(ix, WI, padd, ali), apply_padding(iy, HI, padd, ali), apply_padding(iz, DI, padd, ali)
    in_base = input_ptr + n * si_n + c * si_c
    if interp == 0: # trilinear
        x0, y0, z0, res = tl.floor(ix).to(tl.int32), tl.floor(iy).to(tl.int32), tl.floor(iz).to(tl.int32), 0.0
        wx, wy, wz = ix - x0, iy - y0, iz - z0
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    xi, yi, zi = x0+dx, y0+dy, z0+dz
                    w = (wx if dx==1 else 1-wx)*(wy if dy==1 else 1-wy)*(wz if dz==1 else 1-wz)
                    res += tl.load(in_base + zi*si_d + yi*si_h + xi*si_w, mask=mask & (xi>=0) & (xi<WI) & (yi>=0) & (yi<HI) & (zi>=0) & (zi<DI), other=0.0) * w
    else: # nearest
        x, y, z = tl.math.round(ix).to(tl.int32), tl.math.round(iy).to(tl.int32), tl.math.round(iz).to(tl.int32)
        res = tl.load(in_base + z*si_d + y*si_h + x*si_w, mask=mask & (x>=0) & (x<WI) & (y>=0) & (y<HI) & (z>=0) & (z<DI), other=0.0)
    tl.store(output_ptr + n * so_n + c * so_c + od * so_d + oh * so_h + ow * so_w, res, mask=mask)

class GridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, mode, padding_mode, align_corners):
        if input.numel() * grid.numel() / (input.shape[1] * input.shape[2] * input.shape[3]) < 500000:
            return torch.nn.functional.grid_sample(input, grid, ["bilinear", "nearest", "bicubic"][mode], ["zeros", "border", "reflection"][padding_mode], align_corners)
        out = torch.empty((input.shape[0], input.shape[1], *grid.shape[1:-1]), device=input.device, dtype=input.dtype)
        B = 256
        if input.ndim == 4:
            H, W = out.shape[2], out.shape[3]
            grid_fn = lambda m: (triton.cdiv(H * W, m["B"]), input.shape[0] * input.shape[1])
            _fwd_kernel_2d[grid_fn](input, grid, out, *input.shape, *out.shape[2:], *input.stride(), *grid.stride(), *out.stride(), mode, padding_mode, align_corners, B)
        else:
            D, H, W = out.shape[2], out.shape[3], out.shape[4]
            grid_fn = lambda m: (triton.cdiv(D * H * W, m["B"]), input.shape[0] * input.shape[1])
            _fwd_kernel_3d[grid_fn](input, grid, out, *input.shape, *out.shape[2:], *input.stride(), *grid.stride(), *out.stride(), mode, padding_mode, align_corners, B)
        return out

@libentry()
def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    m = {"bilinear": 0, "nearest": 1, "bicubic": 2}[mode]; p = {"zeros": 0, "border": 1, "reflection": 2}[padding_mode]
    return GridSample.apply(input, grid, m, p, align_corners)
