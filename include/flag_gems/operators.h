#pragma once
#include <optional>
#include "torch/torch.h"

namespace flag_gems {
at::Tensor &exponential_(at::Tensor &self,
                         double lambd = 1.0,
                         c10::optional<at::Generator> gen = c10::nullopt);
at::Tensor zeros(at::IntArrayRef size,
                 c10::optional<at::ScalarType> dtype = ::std::nullopt,
                 c10::optional<at::Layout> layout = ::std::nullopt,
                 c10::optional<at::Device> device = ::std::nullopt,
                 c10::optional<bool> pin_memory = ::std::nullopt);
at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_);
at::Tensor mm_tensor(const at::Tensor &mat1, const at::Tensor &mat2);
at::Tensor &mm_out_tensor(const at::Tensor &mat1, const at::Tensor &mat2, at::Tensor &out);
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim = false,
                   ::std::optional<at::ScalarType> dtype = ::std::nullopt);
at::Tensor sum(const at::Tensor &self, ::std::optional<at::ScalarType> dtype = ::std::nullopt);
std::tuple<at::Tensor, at::Tensor> max_dim(const at::Tensor &self, int64_t dim, bool keepdim);
std::tuple<at::Tensor &, at::Tensor &> max_dim_max(
    const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &out_value, at::Tensor &out_index);
at::Tensor max(const at::Tensor &self);
at::Tensor rms_norm(const at::Tensor &input, const at::Tensor &weight, double epsilon = 1e-5);
void fused_add_rms_norm(at::Tensor &input,
                        at::Tensor &residual,
                        const at::Tensor &weight,
                        double epsilon = 1e-5);
at::Tensor addmm(const at::Tensor &self,
                 const at::Tensor &mat1,
                 const at::Tensor &mat2,
                 const at::Scalar &beta = 1,
                 const at::Scalar &alpha = 1);
at::Tensor &addmm_out(const at::Tensor &self,
                      const at::Tensor &mat1,
                      const at::Tensor &mat2,
                      const at::Scalar &beta,
                      const at::Scalar &alpha,
                      at::Tensor &out);
at::Tensor nonzero(const at::Tensor &inp);
// Rotary embedding
void rotary_embedding_inplace(at::Tensor &q,
                              at::Tensor &k,
                              const at::Tensor &cos,
                              const at::Tensor &sin,
                              const std::optional<at::Tensor> &position_ids = std::nullopt,
                              bool rotary_interleaved = false);
std::tuple<at::Tensor, at::Tensor> rotary_embedding(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const std::optional<at::Tensor> &position_ids = std::nullopt,
    bool rotary_interleaved = false);
std::tuple<at::Tensor, at::Tensor> topk(
    const at::Tensor &x, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true);
at::Tensor contiguous(const at::Tensor &self, at::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);
at::Tensor cat(const at::TensorList &tensors, int64_t dim = 0);
at::Tensor bmm(const at::Tensor &A, const at::Tensor &B);
at::Tensor embedding(const at::Tensor &weight,
                     const at::Tensor &indices,
                     int64_t padding_idx = -1,
                     bool scale_grad_by_freq = false,
                     bool sparse = false);
at::Tensor embedding_backward(const at::Tensor &grad_outputs,
                              const at::Tensor &indices,
                              int64_t num_weights,
                              int64_t padding_idx = -1,
                              bool scale_grad_by_freq = false,
                              bool sparse = false);
at::Tensor argmax(const at::Tensor &self, std::optional<int64_t> dim = std::nullopt, bool keepdim = false);

at::Tensor fill_scalar(const at::Tensor &input, const c10::Scalar &value);

at::Tensor fill_tensor(const at::Tensor &input, const at::Tensor &value);

at::Tensor &fill_scalar_(at::Tensor &input, const c10::Scalar &value);

at::Tensor &fill_tensor_(at::Tensor &input, const at::Tensor &value);

at::Tensor softmax(const at::Tensor &input, int64_t dim, bool half_to_float);

at::Tensor softmax_backward(const at::Tensor &grad_output,
                            const at::Tensor &output,
                            int64_t dim,
                            at::ScalarType input_dtype);

void reshape_and_cache_flash(const at::Tensor &key,
                             const at::Tensor &value,
                             at::Tensor &key_cache,
                             at::Tensor &value_cache,
                             const at::Tensor &slot_mapping,
                             const std::string &kv_cache_dtype,
                             const std::optional<at::Tensor> &k_scale,
                             const std::optional<at::Tensor> &v_scale);

std::tuple<at::Tensor, at::Tensor> flash_attn_varlen_func(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    int64_t max_seqlen_q,
    const at::Tensor &cu_seqlens_q,
    int64_t max_seqlen_k,
    const std::optional<at::Tensor> &cu_seqlens_k = std::nullopt,
    const std::optional<at::Tensor> &seqused_k = std::nullopt,
    const std::optional<at::Tensor> &q_v = std::nullopt,
    double dropout_p = 0.0,
    const std::optional<double> &softmax_scale = std::nullopt,
    bool causal = false,
    c10::optional<at::IntArrayRef> window_size = std::nullopt,
    double softcap = 0.0,
    const std::optional<at::Tensor> &alibi_slopes = std::nullopt,
    bool deterministic = false,
    bool return_attn_probs = false,
    const std::optional<at::Tensor> &block_table = std::nullopt,
    bool return_softmax_lse = false,
    std::optional<at::Tensor> out = std::nullopt,
    const std::optional<at::Tensor> &scheduler_metadata = std::nullopt,
    const std::optional<double> &q_descale = std::nullopt,
    const std::optional<double> &k_descale = std::nullopt,
    const std::optional<double> &v_descale = std::nullopt,
    int64_t fa_version = 2);

struct FlashFwdParams {
  // tensor pointers
  at::Tensor q;
  at::Tensor k;
  at::Tensor v;
  at::Tensor out;
  at::Tensor p;
  at::Tensor lse;
  // strides
  int64_t q_row_stride;
  int64_t k_row_stride;
  int64_t v_row_stride;
  int64_t q_head_stride;
  int64_t k_head_stride;
  int64_t v_head_stride;
  int64_t o_row_stride;
  int64_t o_head_stride;
  // batch strides
  int64_t q_batch_stride;
  int64_t k_batch_stride;
  int64_t v_batch_stride;
  int64_t o_batch_stride;
  // cu_seqlens / seqused_k flags & tensors
  bool is_cu_seqlens_q;
  at::Tensor cu_seqlens_q;
  bool is_cu_seqlens_k;
  at::Tensor cu_seqlens_k;
  bool is_seqused_k;
  at::Tensor seqused_k;
  // sizes
  int64_t batch_size;
  int64_t k_batch_size;
  int64_t num_heads;
  int64_t num_heads_k;
  int64_t h_hk_ratio;
  int64_t seqlen_q;
  int64_t seqlen_k;
  int64_t seqlen_q_rounded;
  int64_t seqlen_k_rounded;
  int64_t head_size;
  int64_t head_size_rounded;
  // scaling factors
  bool is_softcap;
  double softcap;
  double scale_softmax;
  double scale_softmax_log2e;
  // dropout
  bool is_dropout;
  double p_dropout;
  double rp_dropout;
  int64_t p_dropout_in_uint8_t;
  at::Tensor philox_args;
  bool return_softmax;
  // causal & sliding window attention
  bool is_causal;
  bool is_local;
  int64_t window_size_left;
  int64_t window_size_right;
  bool seqlenq_ngroups_swapped;
  // alibi
  bool is_alibi;
  at::Tensor alibi_slopes;
  int64_t alibi_slopes_batch_stride;
  // block table params
  int64_t total_q;
  at::Tensor page_table;
  int64_t page_table_batch_stride;
  int64_t block_size;
};
at::Tensor rwkv_mm_sparsity(const at::Tensor &k, const at::Tensor &v);

std::tuple<at::Tensor, at::Tensor, at::Tensor> rwkv_ka_fusion(const at::Tensor &k,
                                                              const at::Tensor &kk,
                                                              const at::Tensor &a,
                                                              const at::Tensor &ka,
                                                              int64_t H,
                                                              int64_t N);

}  // namespace flag_gems
