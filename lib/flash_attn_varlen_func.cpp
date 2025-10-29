#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <limits>
#include <tuple>
#include "c10/cuda/CUDAStream.h"
#include "c10/util/Optional.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlan_fwd_internal(const at::Tensor& q,
                        const at::Tensor& k,
                        const at::Tensor& v,
                        const at::Tensor& out,
                        const at::Tensor& cu_seqlens_q,
                        const at::Tensor& cu_seqlens_k,
                        const at::Tensor& seqused_k,
                        const at::Tensor& leftpad_k,
                        const at::Tensor& page_table,
                        const at::Tensor& alibi_slopes,
                        int64_t max_seqlen_q,
                        int64_t max_seqlen_k,
                        double p_dropout,
                        double softmax_scale,
                        bool zero_tensors,
                        bool is_causal,
                        int64_t window_size_left,
                        int64_t window_size_right,
                        double softcap,
                        bool return_softmax,
                        const at::Tensor& gen) {
  TORCH_CHECK(q.device() == k.device() && k.device() == v.device(), "q, k, v must be on the same device");
  auto q_device = q.device();
  auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, v must have the same data type");

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");

  TORCH_CHECK(cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_k must be int32");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");

  TORCH_CHECK(page_table.defined(), "page_table must be provided");
  // # q shape: [total_q_tokens, num_heads, head_size]
  // # k shape:
  // #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
  // # batch_size, number of sentences

  const auto total_q = q.size(0);
  const auto num_heads = q.size(1);
  const auto head_size = q.size(2);

  const auto num_heads_k = k.size(2);
  const auto batch_size = cu_seqlens_q.numel() - 1;
  const auto block_size = k.size(1);
  const auto num_pages = k.size(0);
  const auto k_batch_size = num_pages;

  const auto page_table_batch_stride = page_table.stride(0);
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must equal size");
  TORCH_CHECK(cu_seqlens_q.numel() == (batch_size + 1), "cu_seqlens_q must equal size batch_size + 1");
  TORCH_CHECK(cu_seqlens_k.numel() == (batch_size + 1), "cu_seqlens_k must equal size batch_size + 1");
  //  Check output shape
  if (out.defined()) {
    TORCH_CHECK(out.stride(out.dim() - 1) == 1, "Output tensor must have contiguous last dimension");
    TORCH_CHECK(out.scalar_type() == q_dtype, "Output tensor must have the same dtype as input");
    TORCH_CHECK(out.size(0) == total_q && out.size(1) == num_heads && out.size(2) == head_size,
                "Output tensor has incorrect shape");
  }
  if (seqused_k.defined()) {
    TORCH_CHECK(seqused_k.is_contiguous(), "seqused_k must be contiguous");
    TORCH_CHECK(seqused_k.numel() == batch_size, "seqused_k has incorrect size");
  }

  auto q_final = q;
  auto cu_seqlens_q_final = cu_seqlens_q;
  auto num_heads_final = num_heads;
  auto total_q_final = total_q;
  int64_t q_batch_stride = 0;
  auto k_batch_stride = k.stride(0);
  auto v_batch_stride = v.stride(0);
  int64_t o_batch_stride = 0;

  bool is_local = false;
  auto final_is_causal = is_causal;
  if (max_seqlen_q == 1 && !alibi_slopes.defined()) {
    final_is_causal = false;
  }
  if (final_is_causal) {
    window_size_right = 0;
  }
  // check disable swa
  if (window_size_left >= max_seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_k) {
    window_size_right = -1;
  }
  is_local = window_size_left >= 0;

  // Optimize all single-query sequences by swapping the query-group and sequence dimensions
  // Reshape tensor to align Q heads count with K heads count.
  auto seqlenq_ngroups_swapped =
      (max_seqlen_q == 1 && !alibi_slopes.defined() && num_heads_final > num_heads_k &&
       window_size_left < 0 && window_size_right < 0 && p_dropout == 0);
  auto q_groups = num_heads_final / num_heads_k;
  if (seqlenq_ngroups_swapped) {
    q_final = q.reshape({batch_size, num_heads_k, q_groups, head_size})
                  .transpose(1, 2)
                  .reshape({batch_size * q_groups, num_heads_k, head_size});
    max_seqlen_q = q_groups;
    num_heads_final = num_heads_k;
    cu_seqlens_q_final = at::Tensor();

    // q.stride(0) * max_seqlen_q =
    // = (num_heads_k * head_size) * q_groups
    // = (num_heads_k * q_groups) * head_size
    // = num_heads * head_size
    q_batch_stride = q_final.stride(0) * max_seqlen_q;
    k_batch_stride = k.stride(0);
    v_batch_stride = v.stride(0);
  } else {
    q_batch_stride = 0;
    k_batch_stride = 0;
    v_batch_stride = 0;
    o_batch_stride = 0;
  }
  total_q_final = q_final.size(0);
  TORCH_CHECK(!leftpad_k.defined(), "leftpad_k is not supported.");
  TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(num_heads_final % num_heads_k == 0,
              "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(q_final.sizes() == c10::IntArrayRef({total_q_final, num_heads_final, head_size}),
              "q sizes check failed");
  TORCH_CHECK(k.sizes() == c10::IntArrayRef({num_pages, block_size, num_heads_k, head_size}),
              "k sizes check failed");
  TORCH_CHECK(v.sizes() == c10::IntArrayRef({num_pages, block_size, num_heads_k, head_size}),
              "v sizes check failed");
  TORCH_CHECK(k.strides() == v.strides(), "k and v must have the same stride");

  if (softcap > 0.0) {
    TORCH_CHECK(p_dropout == 0, "dropout is not supported if softcap is used.");
  }
  // data preprocess and alignment
  auto round_multiple = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
  auto head_size_rounded = head_size < 192 ? round_multiple(head_size, 32) : 256;
  auto seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  auto seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

  constexpr double LOG2E = 1.4426950408889634074;
  bool is_softcap = false;
  double adjusted_scale_softmax = 0.0;
  double adjusted_softcap = 0.0;
  double adjusted_scale_softmax_log2e = 0.0;
  if (softcap > 0.0) {
    is_softcap = true;
    adjusted_scale_softmax = softcap;
    adjusted_softcap = softmax_scale / softcap;
    adjusted_scale_softmax_log2e = softcap * LOG2E;
  } else {
    is_softcap = false;
    adjusted_softcap = 0.0;
    adjusted_scale_softmax = softmax_scale;
    adjusted_scale_softmax_log2e = softmax_scale * LOG2E;
  }
  // Set alibi params
  bool is_alibi = false;
  int64_t alibi_slopes_batch_stride = 0;
  if (alibi_slopes.defined()) {
    TORCH_CHECK(alibi_slopes.device() == q_device);
    TORCH_CHECK(alibi_slopes.scalar_type() == at::kFloat);
    TORCH_CHECK(alibi_slopes.stride(alibi_slopes.dim() - 1) == 1);
    TORCH_CHECK(alibi_slopes.sizes() == c10::IntArrayRef({
                                            num_heads_final,
                                        }) ||
                alibi_slopes.sizes() == c10::IntArrayRef({
                                            batch_size,
                                            num_heads_final,
                                        }));
    alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    is_alibi = true;
  } else {
    alibi_slopes_batch_stride = 0;
    is_alibi = false;
  }
  at::Tensor out_final = out;
  at::Tensor out_ = at::Tensor();
  at::Tensor lse;
  at::Tensor philox_args;
  at::Tensor p;
  at::Tensor unused;
  {
    const c10::DeviceGuard guard(q_device);
    if (out.defined()) {
      out_ = out;
      if (seqlenq_ngroups_swapped) {
        out_final = out_.view({batch_size, num_heads_k, max_seqlen_q, head_size}).transpose(1, 2);
      } else {
        out_final = out_;
      }
    } else {
      out_ = at::Tensor();
      out_final = at::empty_like(q_final, q_final.options().dtype(v.scalar_type()));
    }
    if (seqlenq_ngroups_swapped) {
      o_batch_stride = out.defined() ? out_.stride(0) : (out_final.stride(0) * max_seqlen_q);
    }
    lse = at::empty({num_heads_final, total_q_final}, at::TensorOptions().dtype(at::kFloat).device(q_device));

    bool is_dropout = false;
    int64_t increment = 0, philox_seed = 0, philox_offset = 0;
    philox_args = at::Tensor();
    if (p_dropout > 0) {
      is_dropout = true;
      increment = batch_size * num_heads_final * 32;

      auto [seed, offset] = flag_gems::philox_backend_seed_offset(increment, c10::nullopt);
      philox_seed = seed;
      philox_offset = offset;
      philox_args =
          at::tensor({philox_seed, philox_offset}, at::TensorOptions().dtype(at::kLong).device(q_device));
    } else {
      is_dropout = false;
      philox_args = at::empty({2}, at::TensorOptions().dtype(at::kLong).device(q_device));
    }
    p_dropout = 1.0 - p_dropout;
    int64_t p_dropout_in_uint8_t = static_cast<int64_t>(std::floor(p_dropout * 255.0));
    double rp_dropout = 1.0 / p_dropout;
    if (return_softmax) {
      TORCH_CHECK(is_dropout, "Only supported with non-zero dropout.");
      p = at::empty({batch_size, num_heads_final, seqlen_q_rounded, seqlen_k_rounded},
                    at::TensorOptions().device(q_device));
    } else {
      p = at::empty({}, at::TensorOptions().device(q_device));
    }
    if (zero_tensors) {
      out_final.zero_();
      lse.fill_(-std::numeric_limits<float>::infinity());
    }

    TORCH_CHECK(q_final.dim() >= 3, "q_final must be at least 3D", q_final.dim());
    TORCH_CHECK(k.dim() >= 3, "k must be at least 3D", k.dim());
    TORCH_CHECK(v.dim() >= 3, "v must be at least 3D", v.dim());
    TORCH_CHECK(out_final.dim() >= 3, "out_final must be at least 3D", out_final.dim());
    const int64_t q_row_stride = q_final.stride(q_final.dim() - 3);
    const int64_t k_row_stride = k.stride(k.dim() - 3);
    const int64_t v_row_stride = v.stride(v.dim() - 3);
    const int64_t q_head_stride = q_final.stride(q_final.dim() - 2);
    const int64_t k_head_stride = k.stride(k.dim() - 2);
    const int64_t v_head_stride = v.stride(v.dim() - 2);
    const int64_t o_row_stride = out_final.stride(out_final.dim() - 3);
    const int64_t o_head_stride = out_final.stride(out_final.dim() - 2);

    const bool is_cu_seqlens_q_flag = cu_seqlens_q_final.defined();
    const bool is_seqused_k_flag = seqused_k.defined();
    const bool is_cu_seqlens_k_flag = !is_seqused_k_flag;

    at::Tensor cu_seqlens_q_safe = is_cu_seqlens_q_flag
                                       ? cu_seqlens_q_final
                                       : at::empty({1}, at::TensorOptions().dtype(at::kInt).device(q_device));
    at::Tensor cu_seqlens_k_safe = cu_seqlens_k;
    at::Tensor seqused_k_safe =
        is_seqused_k_flag ? seqused_k
                          : at::empty({batch_size}, at::TensorOptions().dtype(at::kInt).device(q_device));
    at::Tensor alibi_slopes_safe =
        is_alibi ? alibi_slopes : at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(q_device));

    flag_gems::FlashFwdParams params;
    params.q = q_final;
    params.k = k;
    params.v = v;
    params.out = out_final;
    params.p = p;
    params.lse = lse;
    // strides
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.is_cu_seqlens_q = is_cu_seqlens_q_flag;
    params.cu_seqlens_q = cu_seqlens_q_safe;
    params.is_cu_seqlens_k = is_cu_seqlens_k_flag;
    params.cu_seqlens_k = cu_seqlens_k_safe;
    params.is_seqused_k = is_seqused_k_flag;
    params.seqused_k = seqused_k_safe;
    params.batch_size = batch_size;
    params.k_batch_size = k_batch_size;
    params.num_heads = num_heads_final;
    params.num_heads_k = num_heads_k;
    params.h_hk_ratio = num_heads_final / num_heads_k;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.head_size = head_size;
    params.head_size_rounded = head_size_rounded;
    // scaling（softcap/softmax）
    params.is_softcap = is_softcap;
    params.softcap = adjusted_softcap;
    params.scale_softmax = adjusted_scale_softmax;
    params.scale_softmax_log2e = adjusted_scale_softmax_log2e;
    // dropout
    params.is_dropout = is_dropout;
    params.p_dropout = p_dropout;
    params.rp_dropout = rp_dropout;
    params.p_dropout_in_uint8_t = p_dropout_in_uint8_t;
    params.philox_args = philox_args;
    params.return_softmax = return_softmax;
    params.is_causal = final_is_causal;
    params.is_local = is_local;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
    // alibi
    params.is_alibi = is_alibi;
    params.alibi_slopes = alibi_slopes_safe;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;
    // block table
    params.total_q = total_q_final;
    params.page_table = page_table;
    params.page_table_batch_stride = page_table_batch_stride;
    params.block_size = block_size;

    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceProp prop {};
    cudaGetDeviceProperties(&prop, dev_id);
    int num_sms = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    // We assess which phase the requests are likely to be in and set the config accordingly.
    const double total_rows = static_cast<double>(total_q_final) * static_cast<double>(num_heads_final);
    const double avg_rows_per_sm = total_rows / static_cast<double>(num_sms);
    const double avg_rows_per_batch = static_cast<double>(total_q_final) / static_cast<double>(batch_size);
    const double avg_rows_per_cta = std::min(avg_rows_per_batch, avg_rows_per_sm);
    // Heuristic: if avg_rows_per_sm >= 128, we are likely in prefill phase.
    // This is a rough heuristic and may not be accurate for all scenarios.
    int64_t BLOCK_M = 16;
    int64_t BLOCK_N = 64;
    if (avg_rows_per_cta > 64.0) {
      BLOCK_M = 128;
      BLOCK_N = 32;
    } else if (avg_rows_per_cta > 32.0) {
      BLOCK_M = 64;
      BLOCK_N = 64;
    } else if (avg_rows_per_cta > 16.0) {
      BLOCK_M = 32;
      BLOCK_N = 64;
    } else {
      BLOCK_M = 16;
      BLOCK_N = 64;
    }
    int64_t num_warps = 4;
    int64_t num_stages = 3;

    const unsigned grid_x = static_cast<unsigned>(flag_gems::utils::cdiv(max_seqlen_q, BLOCK_M));
    const unsigned grid_y = static_cast<unsigned>(batch_size);
    const unsigned grid_z = static_cast<unsigned>(num_heads_final);

    const triton_jit::TritonJITFunction& f = triton_jit::TritonJITFunction::get_instance(
        (flag_gems::utils::get_flag_gems_src_path() / "ops" / "flash_kernel.py").string(),
        "flash_varlen_fwd_kernel");
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = stream.stream();

    f(raw_stream,
      grid_x,
      grid_y,
      grid_z,
      static_cast<unsigned>(num_warps),
      static_cast<unsigned>(num_stages),
      params.q,
      params.k,
      params.v,
      params.out,
      params.p,
      params.lse,
      params.q_row_stride,
      params.k_row_stride,
      params.v_row_stride,
      params.q_head_stride,
      params.k_head_stride,
      params.v_head_stride,
      params.o_row_stride,
      params.o_head_stride,
      params.q_batch_stride,
      params.k_batch_stride,
      params.v_batch_stride,
      params.o_batch_stride,
      params.is_cu_seqlens_q,
      params.cu_seqlens_q,
      params.is_cu_seqlens_k,
      params.cu_seqlens_k,
      params.is_seqused_k,
      params.seqused_k,
      // sizes
      params.batch_size,
      params.k_batch_size,
      params.num_heads,
      params.num_heads_k,
      params.h_hk_ratio,
      params.seqlen_q,
      params.seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.head_size,
      params.head_size_rounded,
      // scaling
      params.is_softcap,
      params.softcap,
      params.scale_softmax,
      params.scale_softmax_log2e,
      // dropout
      params.is_dropout,
      params.p_dropout,
      params.rp_dropout,
      params.p_dropout_in_uint8_t,
      params.philox_args,
      params.return_softmax,
      // causal / local / windows / swap
      params.is_causal,
      params.is_local,
      params.window_size_left,
      params.window_size_right,
      params.seqlenq_ngroups_swapped,
      // alibi
      params.is_alibi,
      params.alibi_slopes,
      params.alibi_slopes_batch_stride,
      // block table
      params.total_q,
      params.page_table,
      params.page_table_batch_stride,
      params.block_size,
      // kernel compile-time config
      BLOCK_M,
      BLOCK_N,
      params.head_size_rounded,
      num_warps,
      num_stages);

    if (seqlenq_ngroups_swapped) {
      if (out_.defined()) {
        out_final = out_;
      } else {
        at::Tensor out_swapped =
            out_final.reshape({batch_size, max_seqlen_q, num_heads_k, head_size}).transpose(1, 2);
        out_final = out_swapped.reshape({batch_size, num_heads_k * max_seqlen_q, head_size});
      }
      lse = lse.reshape({num_heads_k, batch_size, max_seqlen_q})
                .reshape({num_heads_k * max_seqlen_q, batch_size});
      unused = at::empty({}, at::TensorOptions().dtype(at::kLong).device(q_device));
    }
  }
  return std::make_tuple(std::move(out_final),
                         std::move(q_final),
                         std::move(k),
                         std::move(v),
                         std::move(lse),
                         std::move(philox_args),
                         std::move(unused),
                         std::move(p));
}
}  // namespace

namespace flag_gems {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> flash_attn_varlen_func(const at::Tensor& q,
                                                          const at::Tensor& k,
                                                          const at::Tensor& v,
                                                          int64_t max_seqlen_q,
                                                          const at::Tensor& cu_seqlens_q,
                                                          int64_t max_seqlen_k,
                                                          const std::optional<at::Tensor>& cu_seqlens_k,
                                                          const std::optional<at::Tensor>& seqused_k,
                                                          const std::optional<at::Tensor>& q_v,
                                                          double dropout_p,
                                                          const std::optional<double>& softmax_scale,
                                                          bool causal,
                                                          c10::optional<at::IntArrayRef> window_size,
                                                          double softcap,
                                                          const std::optional<at::Tensor>& alibi_slopes,
                                                          bool deterministic,
                                                          bool return_attn_probs,
                                                          const std::optional<at::Tensor>& block_table,
                                                          bool return_softmax_lse,
                                                          std::optional<at::Tensor> out,
                                                          const std::optional<at::Tensor>& scheduler_metadata,
                                                          const std::optional<at::Tensor>& q_descale,
                                                          const std::optional<at::Tensor>& k_descale,
                                                          const std::optional<at::Tensor>& v_descale,
                                                          int64_t fa_version) {
  TORCH_CHECK(cu_seqlens_k.has_value() || seqused_k.has_value(),
              "cu_seqlens_k or seqused_k must be provided");
  TORCH_CHECK(!(cu_seqlens_k.has_value() && seqused_k.has_value()),
              "cu_seqlens_k and seqused_k cannot be provided at the same time");
  TORCH_CHECK(!block_table.has_value() || seqused_k.has_value(),
              "seqused_k must be provided if block_table is provided");

  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  if (window_size.has_value()) {
    TORCH_CHECK(window_size.value().size() == 2, "window_size must be a list of 2 elements");
    window_size_left = window_size.value()[0];
    window_size_right = window_size.value()[1];
  }

  double softmax_scale_val;
  if (!softmax_scale.has_value()) {
    softmax_scale_val = 1.0 / std::sqrt(q.size(q.dim() - 1));
  } else {
    softmax_scale_val = softmax_scale.value();
  }
  auto q_cont = q.contiguous();
  auto k_cont = k.contiguous();
  auto v_cont = v.contiguous();

  at::Tensor dummy_cu_seqlens_k;
  if (!cu_seqlens_k.has_value()) {
    dummy_cu_seqlens_k = at::empty_like(cu_seqlens_q);
  }
  const at::Tensor& cu_seqlens_k_ref = cu_seqlens_k.has_value() ? cu_seqlens_k.value() : dummy_cu_seqlens_k;

  TORCH_CHECK(fa_version == 2, "Only FA2 is implemented");
  // TORCH_CHECK(num_splits == 0, "num_splits > 0 is not implemented in GEMS.");

  const at::Tensor empty_undefined = at::Tensor();
  const at::Tensor& seqused_k_ref = seqused_k.has_value() ? seqused_k.value() : empty_undefined;
  const at::Tensor& block_table_ref = block_table.has_value() ? block_table.value() : empty_undefined;
  const at::Tensor& alibi_slopes_ref = alibi_slopes.has_value() ? alibi_slopes.value() : empty_undefined;
  const at::Tensor& out_ref = out.has_value() ? out.value() : empty_undefined;

  auto outputs = mha_varlan_fwd_internal(q_cont,
                                         k_cont,
                                         v_cont,
                                         out_ref,
                                         cu_seqlens_q,
                                         cu_seqlens_k_ref,
                                         seqused_k_ref,
                                         empty_undefined,
                                         block_table_ref,
                                         alibi_slopes_ref,
                                         max_seqlen_q,
                                         max_seqlen_k,
                                         dropout_p,
                                         softmax_scale_val,
                                         false,
                                         causal,
                                         window_size_left,
                                         window_size_right,
                                         softcap,
                                         return_softmax_lse && dropout_p > 0.0,
                                         empty_undefined  // gen
  );

  auto out_tensor = std::get<0>(outputs);
  auto softmax_lse = std::get<4>(outputs);
  return std::make_tuple(std::move(out_tensor), std::move(softmax_lse));
}

}  // namespace flag_gems
