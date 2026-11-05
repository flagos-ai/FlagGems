# SageAttention

`flag_gems.sage_attention` is dense, noncausal inference attention with explicit
quantization semantics. It supports four-dimensional NHD/HND tensors with equal
heads, head dimension128 and unit final stride. Q/K/V share device and dtype.
Empty dimensions, causal/masked/variable-length attention, LSE and backward are
not implemented. Unsupported contracts fail before quantization/launch.

```python
output = flag_gems.sage_attention(q, k, v, tensor_layout="NHD",
                                 pv_dtype="bf16", implementation="baseline")
# Requires SM90 plus the compatible FlagTree TLE compiler:
output = flag_gems.sage_attention(q, k, v, tensor_layout="NHD",
                                 pv_dtype="fp8", implementation="sm90")
```

The ordinary-Triton baseline uses BF16 inputs, per-thread INT8 Q/K and BF16 P/V
with FP32 accumulation. K is centered in the input dtype before quantization.
Q has32 scales per64-token block; K has4 scales per128-token block. Partial
tiles are masked. No TLE, Gluon, TMA or vendor assembly is imported by this path.

The MTT S5000 override optionally uses TLE for the same BF16 numerical contract.
`implementation="tle"` requires the specified TLE APIs; `auto` can select the
baseline before execution if the optional compiler module is unavailable.
Execution errors are never caught and retried with another kernel.

The SM90 override uses per-thread INT8 Q/K and per-channel E4M3 FP8 V, with the
SM90 Sage `fp32+fp32` accumulation contract. It supports FP16/BF16 inputs and
selects TMA/WGMMA pipelines by sequence length. BF16 PV is a separate contract,
not a fallback for FP8. The backend lives under `_nvidia/hopper`; no Gluon is
required by this Sage implementation.

SM90 references must call `sageattn_qk_int8_pv_fp8_cuda_sm90`, with
`qk_quant_gran="per_thread"` and `pv_accum_dtype="fp32+fp32"`. The similarly
named function without `_sm90` uses different quantization/block geometry and
is not a bitwise oracle for this implementation. Dense SDPA is not the
same-arithmetic quantization reference either.

Run `pytest tests/test_sage_attention.py` for quantization-aware numerical tests,
layout boundaries, input preservation, rejection cases and refreshed graphs.
The SM90 native-reference tests need the separate SageAttention reference
package. Run `pytest benchmark/test_sage_attention.py -s` for full API-call
timings, including quantization and allocation. Hardware acceptance on one
backend does not establish another backend's correctness or model quality.
