import pytest
import torch
import random
import triton
import flag_gems
from flag_gems.fused.FLA.chunk_o import chunk_fwd_o
from flag_gems.fused.FLA.utils import input_guard
device = flag_gems.device

DEVICE_COUNT = flag_gems.runtime.device.device_count
_runtime = flag_gems.runtime

# from flag_gems.tests.accuracy_utils import gems_assert_close

random.seed(42)
torch.manual_seed(42)

def torch_chunk_fwd_o(q, k, v, h, *, chunk_size: int = 64, scale=None):
    def _next_power_of_2(n: int) -> int:
        return 1 << (n - 1).bit_length()
    
    q, k, v, h = q.cpu(), k.cpu(), v.cpu(), h.cpu()
    out_dtype = v.dtype
    B, T_, Hg_, K_ = q.shape
    H_, V_ = v.shape[-2], v.shape[-1]
    BT = min(chunk_size, max(16, _next_power_of_2(T_)))
    NT_ = (T_ + BT - 1) // BT
    scale = float(K_ ** -0.5) if scale is None else scale

    # GQA: map q/k heads to value heads.
    if H_ != Hg_:
        m = torch.arange(H_) // (H_ // Hg_)
        q, k = q[:, :, m, :], k[:, :, m, :]

    # Pad time axis once so all chunks are full BT.
    pad = NT_ * BT - T_
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))

    # Unified chunk layout: [B, H, NT, BT, *]
    q = q.transpose(1, 2).reshape(B, H_, NT_, BT, K_).float()
    k = k.transpose(1, 2).reshape(B, H_, NT_, BT, K_).float()
    v = v.transpose(1, 2).reshape(B, H_, NT_, BT, V_).float()
    h = h.reshape(B, NT_, H_, K_, V_).permute(0, 2, 1, 3, 4).float()

    A = (q @ k.transpose(-1, -2)).tril()                 # [B,H,NT,BT,BT]
    o = (q @ h + A.to(out_dtype).float() @ v.to(out_dtype).float()) * scale
    return o.reshape(B, H_, NT_ * BT, V_)[:, :, :T_, :].transpose(1, 2).to(out_dtype).contiguous()

# pytest -v -s --ref cpu tests/test_FLA/test_chunk_fwd_o.py::test_accuracy_chunk_fwd_o
@pytest.mark.chunk_fwd_o
@pytest.mark.parametrize(
    "shape",
    [ # 1 batch size, 2 sequence length
      # 3v/o head num, 4 q/k head num (GQA: Hg <= H) 
      # 5 key/query dim per head 6 value dim per head
        (1, 128, 8, 8, 64, 64),
        (2, 256, 16, 16, 64, 64),
        (4, 128, 8, 8, 64, 64),
        (2, 128, 8, 8, 64, 64),
        (2, 128, 8, 8, 64, 64),
    ],
)                   
@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_accuracy_chunk_fwd_o(shape, chunk_size, dtype):    

    B, T, H, Hg, K, V = shape
    CHUNK_SIZE = chunk_size
    NT = triton.cdiv(T, CHUNK_SIZE)
    q = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H,  V, device=device, dtype=dtype)
    # h: hidden state.shape = (B*NT, H, K, V)，dtype == q/k .dtype
    h = torch.randn(B * NT, H, K, V, device=device, dtype=dtype)

    o_gpu = chunk_fwd_o(q, k, v, h, chunk_size=CHUNK_SIZE)   

    o_ref = torch_chunk_fwd_o(q, k, v, h, chunk_size=CHUNK_SIZE)

    o_gpu = o_gpu.to("cpu", dtype=torch.float32)
    o_ref = o_ref.to("cpu", dtype=torch.float32)

    # gems_assert_close(o_gpu, o_ref, dtype) #,atol=2e-1
    torch.testing.assert_close(o_gpu, o_ref, rtol=1e-1, atol=1e-4)