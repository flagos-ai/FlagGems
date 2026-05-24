"""Drop-in replacement for nn.Linear using TLE SDOT GEMV decode + torch._int_mm prefill.

Decode (M=1, BF16): BF16 activation → in-kernel quant → INT8 SDOT matmul →
BF16 output, all in one @triton.jit that calls triton-cpu's sdot_gemv_fused_bf16
TLE builtin (which dispatches to the NEON SDOT C runtime).

Prefill (M>1): per-row dynamic INT8 quantization in fp32 Python, then
torch._int_mm (which is hooked by flag_gems _arm/ops/int_mm.py to route
to the Triton SVE2 i8mm kernel), then external dequant.

The class exposes the attributes FusedMLPWrapper (in fused/patch_qwen3_mlp.py)
checks for: _packed, _w_scale, K, N — so when gate/up/down are replaced with
TLEInt8Linear, the MLP patch can fuse them into fused_mlp_bf16.
"""

import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import sdot_gemv_fused_bf16 as _cpu_fused_gemv


@triton.jit
def _tle_fused_bf16_gemv_kernel(
    x_ptr, b_packed_ptr, w_scale_ptr, out_ptr,
    K: tl.constexpr, N: tl.constexpr,
):
    """TLE GEMV: BF16 x [K] @ INT8 packed W [K//4, N//4, 4, 4] → BF16 out [N].

    Performs in-kernel dynamic INT8 quantization of x and dequantization of
    output. Single OMP region, no intermediate tensors.
    """
    _cpu_fused_gemv(x_ptr, b_packed_ptr, w_scale_ptr, out_ptr, K, N)


def pack_weights_sdot(w_kn: torch.Tensor) -> torch.Tensor:
    """Pack row-major [K, N] INT8 weight into SDOT-friendly [K//4, N//4, 4, 4].

    SDOT loads 4 consecutive K-bytes from one lane and broadcasts to 4 N-lanes.
    The packed layout ensures each SDOT tile is contiguous in memory for
    maximum L1 cache efficiency. Requires K%4==0 and N%4==0.
    """
    K, N = w_kn.shape
    if K % 4 != 0 or N % 4 != 0:
        raise ValueError(
            f"pack_weights_sdot requires K%4==0 and N%4==0, got K={K} N={N}"
        )
    return w_kn.reshape(K // 4, 4, N // 4, 4).permute(0, 2, 3, 1).contiguous()


class TLEInt8Linear(torch.nn.Module):
    """nn.Linear replacement with TLE SDOT decode + torch._int_mm prefill.

    Args:
        w_int8:   [N, K] int8 tensor (pre-quantized weight, same layout as nn.Linear's
                  .weight.data attribute but dtype=int8).
        w_scale:  [N] fp32 tensor (per-column weight scales); scalar broadcasted
                  tensors also accepted.

    Required: K % 4 == 0 and N % 4 == 0 (SDOT lane requirement).

    Attributes exposed for downstream fusion passes (e.g. patch_qwen3_mlp):
        _packed:   [K//4, N//4, 4, 4] int8  — for SDOT decode
        _w_int8_kn: [K, N] int8            — for torch._int_mm prefill
        _w_scale:  [N] fp32                 — per-column scale
        K, N:      ints
    """

    def __init__(self, w_int8: torch.Tensor, w_scale: torch.Tensor):
        super().__init__()
        if w_int8.dtype != torch.int8:
            raise TypeError(f"w_int8 must be int8, got {w_int8.dtype}")
        self.N, self.K = w_int8.shape
        w_kn = w_int8.t().contiguous()                      # [K, N]
        self._packed = pack_weights_sdot(w_kn)              # [K//4, N//4, 4, 4]
        self._w_int8_kn = w_kn                              # [K, N] for torch._int_mm
        self._w_scale = w_scale.squeeze().to(torch.float32).contiguous()  # [N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        M = x.numel() // shape[-1]
        if M == 1 and x.dtype == torch.bfloat16:
            # Decode fast path — one TLE SDOT GEMV kernel call
            xc = x.reshape(-1).contiguous()
            out = torch.empty(self.N, dtype=torch.bfloat16)
            _tle_fused_bf16_gemv_kernel[(1,)](
                xc, self._packed, self._w_scale, out,
                K=self.K, N=self.N,
            )
            return out.reshape(*shape[:-1], self.N)

        # Prefill: per-row dynamic INT8 quant → _int_mm → dequant
        xf = x.reshape(-1, self.K).contiguous()
        xf32 = xf.float()
        absmax = xf32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_scale = absmax / 127.0
        x_int8 = (xf32 / x_scale).clamp_(-128, 127).to(torch.int8)
        try:
            out_i32 = torch._int_mm(x_int8, self._w_int8_kn)
            out_f32 = out_i32.float() * x_scale * self._w_scale.unsqueeze(0)
        except Exception:
            # FlagGems _int_mm may fall back to aten::mm with int32 operands,
            # which re-enters FlagGems mm and fails for non-BF16 dtype.
            # Use an fp32 matmul fallback that bypasses that chain.
            w_fp32 = (self._w_int8_kn.to(torch.float32)
                      * self._w_scale.unsqueeze(0))  # [K, N]
            out_f32 = xf32 @ w_fp32  # dynamic quant of x was identity here
        return out_f32.to(torch.bfloat16).reshape(*shape[:-1], self.N)

    def extra_repr(self) -> str:
        return f"in_features={self.K}, out_features={self.N}, dtype=int8"
