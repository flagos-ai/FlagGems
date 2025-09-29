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

at::Tensor rwkv_mm_sparsity(const at::Tensor &k, const at::Tensor &v);

std::tuple<at::Tensor, at::Tensor, at::Tensor> rwkv_ka_fusion(const at::Tensor &k,
                                                              const at::Tensor &kk,
                                                              const at::Tensor &a,
                                                              const at::Tensor &ka,
                                                              int64_t H,
                                                              int64_t N);

}  // namespace flag_gems
