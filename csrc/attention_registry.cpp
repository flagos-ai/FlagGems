#include <torch/extension.h>
#include "flash_mla_prefill_alibi_causal.cuh"
#include "flash_mla_paged_decode_kernel.cuh"
#include "flash_mla_decode_specialized.cuh"
#include "flash_mla_decode_fused_wuv.cuh"
#include "flash_mla_decode_fp8.cuh"
// ======【新增这一行】======
#include "flash_attn_varlen.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// Prefill ALiBi + Causal
m.def("flash_mla_prefill_alibi_causal_bf16",
&launch_flash_mla_prefill_alibi_causal<at::BFloat16, 128, 128, 128, 512>,
"Flash MLA Prefill ALiBi Causal, BLOCK_Q=128 BLOCK_K=128",
py::arg("q"), py::arg("latent_k"), py::arg("latent_v"),
py::arg("cum_seqlens"), py::arg("out_latent"),
py::arg("softmax_scale"), py::arg("alibi_slope"),
py::arg("num_heads"), py::arg("max_seqlen"));

// 通用动态shape decode
m.def("flash_mla_paged_decode", &launch_flash_mla_paged_decode,
"Dynamic shape paged MLA decode",
py::arg("q"), py::arg("latent_k_cache"), py::arg("latent_v_cache"),
py::arg("page_table"), py::arg("seq_lens"), py::arg("out"),
py::arg("softmax_scale"), py::arg("batch_size"), py::arg("num_heads"),
py::arg("max_pages_per_seq"));

// Graph Capture 静态特化
m.def("flash_mla_decode_bs32_h16_page32_bf16", &launch_flash_mla_decode_bs32_h16_page32_bf16,
"Graph specialized bs32 h16 page32 bf16");
m.def("flash_mla_decode_bs32_h16_page64_bf16", &launch_flash_mla_decode_bs32_h16_page64_bf16,
"Graph specialized bs32 h16 page64 bf16");

// W_uv fused decode
m.def("flash_mla_decode_fused_wuv_bf16",
&launch_flash_mla_decode_fused_wuv<at::BFloat16,128,512,64>,
"Decode fuse W_uv projection, eliminate post GEMM");

// FP8 decode
m.def("flash_mla_decode_fp8", &launch_flash_mla_decode_fp8,
"FP8 E4M3 MLA decode with amax dequant");

// ======【从这里开始追加新算子绑定，放在原有全部代码最后】======
m.def("flash_attn_varlen_fun",
&flash_attn_varlen_fun_launcher,
"FlashAttention varlen causal (standard MHA, not MLA), prefill/decode",
py::arg("q"),
py::arg("k"),
py::arg("v"),
py::arg("cum_seqlens_q"),
py::arg("cum_seqlens_k"),
py::arg("softmax_scale"),
py::arg("num_heads"),
py::arg("max_seqlen_q"),
py::arg("max_seqlen_k"),
py::arg("causal") = true);
}
