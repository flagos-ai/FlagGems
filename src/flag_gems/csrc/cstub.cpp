#include <cstdlib>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "torch/python.h"

#include "flag_gems/operators.h"

namespace py = pybind11;

namespace {

c10::Scalar py_object_to_scalar(const py::object &value) {
  if (py::isinstance<py::bool_>(value)) {
    return c10::Scalar(value.cast<bool>());
  }
  if (py::isinstance<py::int_>(value)) {
    return c10::Scalar(value.cast<int64_t>());
  }
  if (py::isinstance<py::float_>(value)) {
    return c10::Scalar(value.cast<double>());
  }
  throw py::type_error("index_fill_scalar_ value must be a bool, int, or float scalar");
}

py::object scalar_to_py_object(const c10::Scalar &value) {
  switch (value.type()) {
    case c10::ScalarType::Bool:
      return py::bool_(value.toBool());
    case c10::ScalarType::Long:
      return py::int_(value.toLong());
    case c10::ScalarType::UInt64:
      return py::int_(value.toUInt64());
    case c10::ScalarType::Double:
      return py::float_(value.toDouble());
    default:
      throw py::type_error(
          "index_fill_scalar_ value must be a bool, int, or float scalar");
  }
}

bool index_fill_uses_device_bounds_check() {
  const char *mode_env = std::getenv("FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK");
  if (mode_env == nullptr) return true;

  const std::string mode(mode_env);
  return mode.empty() || mode == "0" || mode == "false" || mode == "off" ||
         mode == "device" || mode == "default" || mode == "none" ||
         mode == "disable" || mode == "disabled";
}

at::Tensor index_fill_scalar_dispatch(const at::Tensor &input,
                                      int64_t dim,
                                      const at::Tensor &index,
                                      py::object value) {
  if (input.is_contiguous() && index_fill_uses_device_bounds_check()) {
    return flag_gems::index_fill_scalar(
        input, dim, index, py_object_to_scalar(value));
  }

  return py::module_::import("flag_gems.ops.index_fill")
      .attr("index_fill_scalar")(input, dim, index, value)
      .cast<at::Tensor>();
}

at::Tensor &index_fill_scalar_dispatch_(at::Tensor &input,
                                        int64_t dim,
                                        const at::Tensor &index,
                                        py::object value) {
  if (input.is_contiguous() && index_fill_uses_device_bounds_check()) {
    return flag_gems::index_fill_scalar_(
        input, dim, index, py_object_to_scalar(value));
  }

  py::module_::import("flag_gems.ops.index_fill")
      .attr("index_fill_scalar_")(input, dim, index, value);
  return input;
}

at::Tensor index_fill_scalar_aten_dispatch(const at::Tensor &input,
                                           int64_t dim,
                                           const at::Tensor &index,
                                           const c10::Scalar &value) {
  if (input.is_contiguous() && index_fill_uses_device_bounds_check()) {
    return flag_gems::index_fill_scalar(input, dim, index, value);
  }

  py::gil_scoped_acquire gil;
  return index_fill_scalar_dispatch(
      input, dim, index, scalar_to_py_object(value));
}

at::Tensor &index_fill_scalar_aten_dispatch_(at::Tensor &input,
                                             int64_t dim,
                                             const at::Tensor &index,
                                             const c10::Scalar &value) {
  if (input.is_contiguous() && index_fill_uses_device_bounds_check()) {
    return flag_gems::index_fill_scalar_(input, dim, index, value);
  }

  py::gil_scoped_acquire gil;
  return index_fill_scalar_dispatch_(
      input, dim, index, scalar_to_py_object(value));
}

#if defined(FLAGGEMS_USE_CUDA)
class IndexFillAtenRegistration {
 public:
  IndexFillAtenRegistration()
      : library_(std::make_unique<torch::Library>(
            torch::Library::IMPL,
            "aten",
            c10::DispatchKey::CUDA,
            __FILE__,
            __LINE__)) {
    library_->impl("index_fill.int_Scalar",
                   TORCH_FN(index_fill_scalar_aten_dispatch));
    library_->impl("index_fill_.int_Scalar",
                   TORCH_FN(index_fill_scalar_aten_dispatch_));
  }

 private:
  std::unique_ptr<torch::Library> library_;
};
#endif

}  // namespace

// TODO: use pytorch's argparse utilities to generate CPython bindings, since it is more efficient than
// bindings provided by torch library, since it is in a boxed fashion
PYBIND11_MODULE(c_operators, m) {
#ifdef FLAGGEMS_POINTWISE_DYNAMIC
  // add
  m.def(
      "add_tensor",
      [](const at::Tensor &self, const at::Tensor &other, double alpha) {
        return flag_gems::add_tensor(self, other, alpha);
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("alpha") = 1.0);
  m.def(
      "add_scalar",
      [](const at::Tensor &self, const at::Scalar &other, double alpha) {
        return flag_gems::add_scalar(self, other, alpha);
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("alpha") = 1.0);
  m.def(
      "add_tensor_inplace",
      [](at::Tensor &self, const at::Tensor &other, double alpha) {
        return flag_gems::add_tensor_inplace(self, other, alpha);
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("alpha") = 1.0);
  m.def(
      "add_scalar_inplace",
      [](at::Tensor &self, const at::Scalar &other, double alpha) {
        return flag_gems::add_scalar_inplace(self, other, alpha);
      },
      py::arg("self"),
      py::arg("other"),
      py::arg("alpha") = 1.0);
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
  // fill
  m.def("fill.Scalar", &flag_gems::fill_scalar);
  m.def("fill.Tensor", &flag_gems::fill_tensor);
  m.def("fill_.Scalar", &flag_gems::fill_scalar_);
  m.def("fill_.Tensor", &flag_gems::fill_tensor_);
#endif
  m.def("act_quant",
        &flag_gems::act_quant_triton,
        py::arg("x"),
        py::arg("block_size") = 128,
        py::arg("scale_fmt") = py::none());
  m.def("exponential_", &flag_gems::exponential_);
  m.def("addmm", &flag_gems::addmm);
  m.def("mm", &flag_gems::mm_tensor);
  m.def("zeros", &flag_gems::zeros);
  m.def(
      "sum_dim",
      [](const at::Tensor &self,
         const std::optional<std::vector<int64_t>> &dim,
         bool keepdim,
         const std::optional<at::ScalarType> &dtype) {
        at::OptionalIntArrayRef dim_ref =
            dim.has_value() ? at::OptionalIntArrayRef(*dim) : at::OptionalIntArrayRef();
        return flag_gems::sum_dim(self, dim_ref, keepdim, dtype);
      },
      py::arg("self"),
      py::arg("dim") = py::none(),
      py::arg("keepdim") = false,
      py::arg("dtype") = py::none());
  m.def("sum", &flag_gems::sum);
  m.def("max_dim", &flag_gems::max_dim);
  m.def("max", &flag_gems::max);
  m.def("max_dim_max", &flag_gems::max_dim_max);
  m.def("rms_norm", &flag_gems::rms_norm);
  m.def("gemma_rms_norm", &flag_gems::gemma_rms_norm);
  m.def("fused_add_rms_norm", &flag_gems::fused_add_rms_norm);
  m.def("nonzero", &flag_gems::nonzero);
  m.def("rotary_embedding", &flag_gems::rotary_embedding);
  m.def("rotary_embedding_inplace", &flag_gems::rotary_embedding_inplace);
  m.def("topk", &flag_gems::topk);
  m.def(
      "contiguous",
      [](const at::Tensor &self, at::MemoryFormat memory_format) {
        return flag_gems::contiguous(self, memory_format);
      },
      py::arg("self"),
      py::arg("memory_format") = c10::MemoryFormat::Contiguous);
  m.def(
      "cat",
      [](const std::vector<at::Tensor> &tensors, int64_t dim) { return flag_gems::cat(tensors, dim); },
      py::arg("tensors"),
      py::arg("dim") = 0);
  m.def("bmm", &flag_gems::bmm);
  m.def("embedding", &flag_gems::embedding);
  m.def("embedding_backward", &flag_gems::embedding_backward);
  m.def("argmax", &flag_gems::argmax);

  m.def("sort", &flag_gems::sort);
  m.def("sort_stable", &flag_gems::sort_stable);
  m.def("softmax", &flag_gems::softmax);
  m.def("softmax_backward", &flag_gems::softmax_backward);
  m.def("reshape_and_cache_flash", &flag_gems::reshape_and_cache_flash);
  m.def("flash_attn_varlen_func", &flag_gems::flash_attn_varlen_func);
  m.def("rwkv_mm_sparsity", &flag_gems::rwkv_mm_sparsity);
  m.def("rwkv_ka_fusion", &flag_gems::rwkv_ka_fusion);
  m.def("copy_", &flag_gems::copy_);
  m.def("to_copy", &flag_gems::to_copy);
  m.def(
      "index_fill_scalar_raw",
      [](const at::Tensor &input,
         int64_t dim,
         const at::Tensor &index,
         py::object value) {
        c10::Scalar scalar = py_object_to_scalar(value);
        return flag_gems::index_fill_scalar(input, dim, index, scalar);
      });
  m.def(
      "index_fill_scalar_raw_",
      [](at::Tensor &input, int64_t dim, const at::Tensor &index, py::object value)
          -> at::Tensor & {
        c10::Scalar scalar = py_object_to_scalar(value);
        return flag_gems::index_fill_scalar_(input, dim, index, scalar);
      });
  m.def("index_fill_scalar", &index_fill_scalar_dispatch);
  m.def("index_fill_scalar_", &index_fill_scalar_dispatch_);
#if defined(FLAGGEMS_USE_CUDA)
  py::class_<IndexFillAtenRegistration>(m, "IndexFillAtenRegistration")
      .def(py::init<>());
#endif
  m.def("fp8_matmul",
        &flag_gems::fp8_matmul,
        py::arg("a"),
        py::arg("a_s"),
        py::arg("b"),
        py::arg("b_s"),
        py::arg("scale_dtype") = at::kFloat);
  m.def("fp8_matmul_direct",
        &flag_gems::fp8_matmul_direct,
        py::arg("a"),
        py::arg("a_s"),
        py::arg("b"),
        py::arg("b_s"),
        py::arg("scale_dtype") = at::kFloat);
}
namespace flag_gems {
TORCH_LIBRARY(flag_gems, m) {
#ifdef FLAGGEMS_POINTWISE_DYNAMIC
  // add
  m.def("add_tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", {at::Tag::pt2_compliant_tag});
  m.def("add_scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", {at::Tag::pt2_compliant_tag});
  m.def("add_tensor_inplace(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
        {at::Tag::pt2_compliant_tag});
  m.def("add_scalar_inplace(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
        {at::Tag::pt2_compliant_tag});
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
  // fill
  m.def("fill.Scalar(Tensor self, Scalar value) -> Tensor");
  m.def("fill.Tensor(Tensor self, Tensor value) -> Tensor");
  m.def("fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
  m.def("fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)");
#endif
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
  // m.def("add_tensor(Tensor self, Tensor other) -> Tensor", {at::Tag::pt2_compliant_tag});
  // Norm
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("gemma_rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
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
  // // div
  // m.def("div.Tensor(Tensor self, Tensor other) -> Tensor");
  // m.def("div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // m.def("div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor");
  // m.def("div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)");
  // m.def("floor_divide(Tensor self, Tensor other) -> Tensor");
  // m.def("floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // m.def("divide.Tensor(Tensor self, Tensor other) -> Tensor");
  // m.def("divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // m.def("divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor");
  // m.def("divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)");
  // m.def("true_divide.Tensor(Tensor self, Tensor other) -> Tensor");
  // m.def("true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // m.def("remainder.Tensor(Tensor self, Tensor other) -> Tensor");
  // m.def("remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)");
  // sort
  m.def("sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)");
  m.def(
      "sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, "
      "Tensor indices)");

  // m.def("fill.Scalar(Tensor self, Scalar value) -> Tensor");
  // m.def("fill.Tensor(Tensor self, Tensor value) -> Tensor");
  // m.def("fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
  // m.def("fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)");
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

// Define dispatch key based on backend
// CUDA and IX use CUDA dispatch key (IX is CUDA-compatible)
// NPU and MUSA use PrivateUse1 dispatch key
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX) || defined(FLAGGEMS_USE_HCU)
#define FLAGGEMS_DISPATCH_KEY CUDA
#elif defined(FLAGGEMS_USE_NPU) || defined(FLAGGEMS_USE_MUSA) || defined(FLAGGEMS_USE_GCU)
#define FLAGGEMS_DISPATCH_KEY PrivateUse1
#else
#error \
    "No backend defined. Define one of: FLAGGEMS_USE_CUDA, FLAGGEMS_USE_IX, FLAGGEMS_USE_NPU, FLAGGEMS_USE_MUSA, FLAGGEMS_USE_GCU, FLAGGEMS_USE_HCU"
#endif

TORCH_LIBRARY_IMPL(flag_gems, FLAGGEMS_DISPATCH_KEY, m) {
#ifdef FLAGGEMS_POINTWISE_DYNAMIC
  // add
  m.impl("add_tensor", TORCH_FN(add_tensor));
  m.impl("add_scalar", TORCH_FN(add_scalar));
  m.impl("add_tensor_inplace", TORCH_FN(add_tensor_inplace));
  m.impl("add_scalar_inplace", TORCH_FN(add_scalar_inplace));
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
  // fill
  m.impl("fill.Scalar", TORCH_FN(fill_scalar));
  m.impl("fill.Tensor", TORCH_FN(fill_tensor));
  m.impl("fill_.Scalar", TORCH_FN(fill_scalar_));
  m.impl("fill_.Tensor", TORCH_FN(fill_tensor_));
#endif

  m.impl("exponential_", TORCH_FN(exponential_));
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("bmm", TORCH_FN(bmm));
  m.impl("mm", TORCH_FN(mm_tensor));
  m.impl("zeros", TORCH_FN(zeros));
  m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
  m.impl("sum", TORCH_FN(sum));
  m.impl("max.dim_max", TORCH_FN(max_dim_max));
  m.impl("max.dim", TORCH_FN(max_dim));
  m.impl("max", TORCH_FN(max));
  m.impl("rms_norm", TORCH_FN(rms_norm));
  m.impl("gemma_rms_norm", TORCH_FN(gemma_rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(fused_add_rms_norm));
  m.impl("nonzero", TORCH_FN(nonzero));
  m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
  m.impl("rotary_embedding_inplace", TORCH_FN(rotary_embedding_inplace));
  m.impl("topk", TORCH_FN(topk));
  m.impl("contiguous", TORCH_FN(contiguous));
  m.impl("cat", TORCH_FN(cat));

  m.impl("embedding", TORCH_FN(embedding));
  m.impl("embedding_backward", TORCH_FN(embedding_backward));
  m.impl("argmax", TORCH_FN(argmax));

  m.impl("sort", TORCH_FN(sort));
  m.impl("sort.stable", TORCH_FN(sort_stable));

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
