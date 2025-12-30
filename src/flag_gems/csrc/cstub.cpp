#include <pybind11/pybind11.h>
#include "torch/python.h"

#include "flag_gems/operators.h"

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
  m.def("sum_dim", &flag_gems::sum_dim);
  m.def("sum", &flag_gems::sum);
  m.def("max_dim", &flag_gems::max_dim);
  m.def("max", &flag_gems::max);
  m.def("add_tensor", &flag_gems::add_tensor);
  m.def("max_dim_max", &flag_gems::max_dim_max);
  m.def("rms_norm", &flag_gems::rms_norm);
  m.def("fused_add_rms_norm", &flag_gems::fused_add_rms_norm);
  m.def("nonzero", &flag_gems::nonzero);
  // Rotary embedding
  m.def("rotary_embedding", &flag_gems::rotary_embedding);
  m.def("rotary_embedding_inplace", &flag_gems::rotary_embedding_inplace);
  m.def("bmm", &flag_gems::bmm);
  // div
  m.def("div.Tensor", &flag_gems::true_div);
  m.def("div_.Tensor", &flag_gems::true_div_);
  m.def("div.Tensor_mode", &flag_gems::div_mode);
  m.def("div_.Tensor_mode", &flag_gems::div_mode_);
  m.def("floor_divide", &flag_gems::floor_div);
  m.def("floor_divide_.Tensor", &flag_gems::floor_div_);
  m.def("divide.Tensor", &flag_gems::true_div);
  m.def("divide_.Tensor", &flag_gems::true_div_);
  m.def("divide.Tensor_mode", &flag_gems::div_mode);
  m.def("divide_.Tensor_mode", &flag_gems::div_mode_);
  m.def("true_divide.Tensor", &flag_gems::true_div);
  m.def("true_divide_.Tensor", &flag_gems::true_div_);
  m.def("remainder.Tensor", &flag_gems::remainder);
  m.def("remainder_.Tensor", &flag_gems::remainder_);
  m.def("rwkv_mm_sparsity", &flag_gems::rwkv_mm_sparsity);
  m.def("rwkv_ka_fusion", &flag_gems::rwkv_ka_fusion);
  m.def("copy_", &flag_gems::copy_);
  m.def("to_copy", &flag_gems::to_copy);
}
namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
  m.def("exponential_(Tensor(a!) x, float  lambd = 1.0, *,Generator? gen = None) -> Tensor(a!)");
  // blas
  m.def("addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor");
  m.def("mm(Tensor self, Tensor mat2) -> Tensor");

  m.def(
      "zeros(SymInt[] size, ScalarType? dtype=None,Layout? layout=None, Device? device=None, bool? "
      "pin_memory=None) -> Tensor");
  m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
  m.def("sum(Tensor self, *, ScalarType? dtype=None) -> Tensor");
  m.def(
      "max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> "
      "(Tensor(a!) values, Tensor(b!) indices)");
  m.def("max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("max(Tensor self) -> Tensor");
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) -> ()");
  m.def("nonzero(Tensor self) -> Tensor");
  // rotary_embedding
  m.def(
      "rotary_embedding_inplace(Tensor! q, Tensor! k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> ()");
  m.def(
      "rotary_embedding(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor? position_ids=None, "
      "bool rotary_interleaved=False) -> (Tensor, Tensor)");  // q and k may be view to other size
  m.def("topk(Tensor x, SymInt k, int dim, bool largest, bool sorted) -> (Tensor, Tensor)");
  m.def("contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)");
  m.def("cat(Tensor[] tensors, int dim=0) -> Tensor");
  m.def("bmm(Tensor self, Tensor mat2) -> Tensor");
  m.def(
      "embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool "
      "sparse=False) -> Tensor");
  m.def(
      "embedding_backward(Tensor grad_outputs, Tensor indices, SymInt num_weights, SymInt padding_idx, bool "
      "scale_grad_by_freq, bool sparse) -> Tensor");
  m.def("argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor");
  // div
  m.def("div.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor");
  m.def("div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)");
  m.def("floor_divide(Tensor self, Tensor other) -> Tensor");
  m.def("floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("divide.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor");
  m.def("divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)");
  m.def("true_divide.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  m.def("remainder.Tensor(Tensor self, Tensor other) -> Tensor");
  m.def("remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // sort
  m.def("sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)");
  m.def(
      "sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, "
      "Tensor indices)");

  m.def("fill.Scalar(Tensor self, Scalar value) -> Tensor");
  m.def("fill.Tensor(Tensor self, Tensor value) -> Tensor");
  m.def("fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
  m.def("fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)");
  m.def("softmax(Tensor input, int dim, bool half_to_float=False) -> Tensor");
  m.def("softmax_backward(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor");
  m.def(
      "reshape_and_cache_flash(Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, "
      "Tensor slot_mapping, str kv_cache_dtype, Tensor? k_scale=None, Tensor? v_scale=None) -> "
      "()");
  m.def(
      "flash_attn_varlen_func(Tensor q, Tensor k, Tensor v, SymInt max_seqlen_q, Tensor cu_seqlens_q, SymInt "
      "max_seqlen_k, "
      "Tensor? cu_seqlens_k=None, Tensor? seqused_k=None, Tensor? q_v=None, float dropout_p=0.0, float? "
      "softmax_scale=None, "
      "bool causal=False, SymInt[]? window_size=None,float softcap=0.0, "
      "Tensor? alibi_slopes=None, "
      "bool deterministic=False, bool return_attn_probs=False, Tensor? block_table=None, bool "
      "return_softmax_lse=False, "
      "Tensor? out=None, Tensor? scheduler_metadata=None, Tensor? q_descale=None, Tensor? k_descale=None, "
      "Tensor? v_descale=None, Tensor? s_aux=None, SymInt num_splits=0, SymInt cp_world_size=1, "
      "SymInt cp_rank=0, Tensor? cp_tot_seqused_k=None, SymInt fa_version=2) -> (Tensor, Tensor)");

  m.def("rwkv_mm_sparsity(Tensor k, Tensor v) -> Tensor");
  m.def("rwkv_ka_fusion(Tensor k, Tensor kk, Tensor a, Tensor ka, int H, int N) -> (Tensor, Tensor, Tensor)");
  m.def("copy_(Tensor(a!) dst, Tensor src, bool non_blocking=False) -> Tensor(a!)");
  m.def(
      "to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
      "pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(flag_gems, CUDA, m) {
  m.impl("exponential_", TORCH_FN(exponential_));
  // blas
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("bmm", TORCH_FN(bmm));
  m.impl("mm", TORCH_FN(mm_tensor));

  m.impl("zeros", TORCH_FN(zeros));
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("sum", TORCH_FN(sum));
  m.impl("max.dim_max", TORCH_FN(max_dim_max));
  m.impl("max.dim", TORCH_FN(max_dim));
  m.impl("max", TORCH_FN(max));
  m.impl("add_tensor", TORCH_FN(add_tensor));
  // Norm
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
  m.impl("nonzero", TORCH_FN(nonzero));
  // Rotary embedding
  m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(rotary_embedding_inplace));
  m.impl("topk", TORCH_FN(topk));
  m.impl("contiguous", TORCH_FN(contiguous));
  m.impl("cat", TORCH_FN(cat));

  m.impl("embedding", TORCH_FN(embedding));
  m.impl("embedding_backward", TORCH_FN(embedding_backward));
  m.impl("argmax", TORCH_FN(argmax));
  // div
  m.impl("div.Tensor", TORCH_FN(true_div));
  m.impl("div_.Tensor", TORCH_FN(true_div_));
  m.impl("div.Tensor_mode", TORCH_FN(div_mode));
  m.impl("div_.Tensor_mode", TORCH_FN(div_mode_));
  m.impl("div.Scalar", TORCH_FN(true_div));
  m.impl("div_.Scalar", TORCH_FN(true_div_));
  m.impl("div.Scalar_mode", TORCH_FN(div_mode));
  m.impl("div_.Scalar_mode", TORCH_FN(div_mode_));
  m.impl("floor_divide", TORCH_FN(floor_div));
  m.impl("floor_divide_.Tensor", TORCH_FN(floor_div_));
  m.impl("floor_divide.Scalar", TORCH_FN(floor_div));
  m.impl("floor_divide_.Scalar", TORCH_FN(floor_div_));
  m.impl("divide.Tensor", TORCH_FN(true_div));
  m.impl("divide_.Tensor", TORCH_FN(true_div_));
  m.impl("divide.Scalar", TORCH_FN(true_div));
  m.impl("divide_.Scalar", TORCH_FN(true_div_));
  m.impl("divide.Tensor_mode", TORCH_FN(div_mode));
  m.impl("divide_.Tensor_mode", TORCH_FN(div_mode_));
  m.impl("divide.Scalar_mode", TORCH_FN(div_mode));
  m.impl("divide_.Scalar_mode", TORCH_FN(div_mode_));
  m.impl("true_divide.Tensor", TORCH_FN(true_div));
  m.impl("true_divide_.Tensor", TORCH_FN(true_div_));
  m.impl("remainder.Scalar", TORCH_FN(remainder));
  m.impl("remainder_.Scalar", TORCH_FN(remainder_));
  m.impl("remainder.Tensor", TORCH_FN(remainder));
  m.impl("remainder_.Tensor", TORCH_FN(remainder_));
  m.impl("remainder.Scalar_Tensor", TORCH_FN(remainder));
  // sort
  m.impl("sort", TORCH_FN(sort));
  m.impl("sort.stable", TORCH_FN(sort_stable));

  m.impl("fill.Scalar", TORCH_FN(fill_scalar));
  m.impl("fill.Tensor", TORCH_FN(fill_tensor));
  m.impl("fill_.Scalar", TORCH_FN(fill_scalar_));
  m.impl("fill_.Tensor", TORCH_FN(fill_tensor_));
  m.impl("softmax", TORCH_FN(softmax));
  m.impl("softmax_backward", TORCH_FN(softmax_backward));
  m.impl("reshape_and_cache_flash", TORCH_FN(reshape_and_cache_flash));
  m.impl("flash_attn_varlen_func", TORCH_FN(flash_attn_varlen_func));
  m.impl("rwkv_mm_sparsity", TORCH_FN(rwkv_mm_sparsity));
  m.impl("rwkv_ka_fusion", TORCH_FN(rwkv_ka_fusion));
  m.impl("to_copy", TORCH_FN(to_copy));
  m.impl("copy_", TORCH_FN(copy_));
}
}  // namespace flag_gems
