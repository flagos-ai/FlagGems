#pragma once

// ==========================================================================
// C++ implementation of pointwise_dynamic argument preparation and dispatch.
//
// This file mirrors the logic in the Python version:
//   flag_gems/utils/pointwise_dynamic.py — PointwiseDynamicFunction
//     - prepare_args(): broadcasting, fast-path detection, strided views
//     - __call__(): dtype promotion, output allocation, kernel launch
//
// If the Python codegen or dispatch logic changes, this file should be
// updated accordingly to keep the two paths consistent.
//
// NOTE: This file is NOT auto-generated.  The generated headers
// (pointwise_manifest.h, pointwise_runtime.h) provide per-operator
// registry data and thin wrapper functions.
// ==========================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "c10/cuda/CUDAStream.h"
#include "pointwise_manifest.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace pointwise_dynamic {

using namespace triton_jit;

// ==========================================================================
// Shape utilities
// ==========================================================================

inline c10::SmallVector<int64_t, 8> broadcast_shapes(
    const c10::SmallVector<c10::SmallVector<int64_t, 8>, 4>& shapes) {
  if (shapes.empty()) return {};
  size_t max_ndim = 0;
  for (const auto& shape : shapes) {
    max_ndim = std::max(max_ndim, shape.size());
  }
  c10::SmallVector<int64_t, 8> result(max_ndim, 1);
  for (const auto& shape : shapes) {
    size_t offset = max_ndim - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t dim = shape[i];
      int64_t& out_dim = result[offset + i];
      if (out_dim == 1) {
        out_dim = dim;
      } else if (dim != 1 && dim != out_dim) {
        throw std::runtime_error("Shapes cannot be broadcast together");
      }
    }
  }
  return result;
}

inline c10::SmallVector<int64_t, 8> broadcasted_stride(at::IntArrayRef shape,
                                                       at::IntArrayRef stride,
                                                       const c10::SmallVector<int64_t, 8>& target_shape) {
  size_t ndim = target_shape.size();
  size_t offset = ndim - shape.size();
  c10::SmallVector<int64_t, 8> result(ndim, 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == target_shape[offset + i]) {
      result[offset + i] = stride[i];
    }
  }
  return result;
}

// ==========================================================================
// Stride order computation (for block pointer kernels)
// Sorts dimension indices by ascending absolute stride value.
// ==========================================================================

inline c10::SmallVector<int64_t, 8> compute_stride_order(at::IntArrayRef strides) {
  c10::SmallVector<int64_t, 8> order(strides.size());
  std::iota(order.begin(), order.end(), int64_t {0});
  std::sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
    return std::llabs(strides[lhs]) < std::llabs(strides[rhs]);
  });
  return order;
}

// ==========================================================================
// Launch heuristics
// ==========================================================================

inline int64_t next_power_of_2(int64_t n) {
  if (n <= 1) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

// 1D heuristic: used for fast-path (1D tile) kernels
inline int64_t heuristics_for_tile_size(int64_t num_tasks) {
  int64_t tile = 1;
  while (tile < 1024 && tile < num_tasks) tile *= 2;
  return std::min(tile, int64_t(1024));
}

// ND heuristic: budget-based per-dimension tile sizes.
// Mirrors Python's heuristics_for_tile_size(max_tile_size, *shape) in
// flag_gems/utils/shape_utils.py.  Distributes the tile budget starting
// from the innermost (last) dimension.
inline c10::SmallVector<int64_t, 8> heuristics_for_nd_tile_sizes(int64_t max_tile_size,
                                                                 const c10::SmallVector<int64_t, 8>& shape) {
  int ndim = static_cast<int>(shape.size());
  c10::SmallVector<int64_t, 8> tile_sizes(ndim, 0);
  for (int i = ndim - 1; i >= 0; --i) {
    int64_t ts = std::min(max_tile_size, next_power_of_2(shape[i]));
    tile_sizes[i] = ts;
    max_tile_size = std::max(int64_t(1), max_tile_size / ts);
  }
  return tile_sizes;
}

inline int heuristics_for_num_warps(int64_t tile_size) {
  if (tile_size <= 256) return 1;
  if (tile_size <= 512) return 2;
  if (tile_size <= 1024) return 4;
  return 8;
}

// ==========================================================================
// Fast-path detection (same logic as Python pointwise_dynamic)
// ==========================================================================

inline bool all_same_shape(const c10::SmallVector<at::Tensor, 4>& tensors) {
  if (tensors.empty()) return true;
  const auto& first_sizes = tensors[0].sizes();
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[i].sizes() != first_sizes) {
      return false;
    }
  }
  return true;
}

inline bool all_contiguous(const c10::SmallVector<at::Tensor, 4>& tensors) {
  for (const auto& t : tensors) {
    if (!t.is_contiguous()) {
      return false;
    }
  }
  return true;
}

inline bool all_same_stride(const c10::SmallVector<at::Tensor, 4>& tensors) {
  if (tensors.empty()) return true;
  const auto& first_strides = tensors[0].strides();
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[i].strides() != first_strides) {
      return false;
    }
  }
  return true;
}

// Fast path: when all tensors have same shape and are contiguous (or same
// stride + non-overlapping dense), we can collapse to 1D for simpler indexing.
inline bool use_fast_path(const c10::SmallVector<at::Tensor, 4>& tensors) {
  if (!all_same_shape(tensors)) {
    return false;
  }
  if (all_contiguous(tensors)) {
    return true;
  }
  if (all_same_stride(tensors) && !tensors.empty() && tensors[0].is_non_overlapping_and_dense()) {
    return true;
  }
  return false;
}

// ==========================================================================
// Internal overlap check (matches Python has_internal_overlapping)
// ==========================================================================

inline bool has_internal_overlapping(const at::Tensor& t) {
  if (t.numel() <= 1) return false;
  // Match Python: at::has_internal_overlap returns MemOverlap::Yes for actual
  // overlap. Using !is_non_overlapping_and_dense() is too strict — it rejects
  // non-dense views (e.g. KV cache slices) that have no overlap.
  return at::has_internal_overlap(t) == at::MemOverlap::Yes;
}

// ==========================================================================
// Dtype promotion (mirrors Python's elementwise_dtypes logic)
// ==========================================================================

// Dtype classification and conversion — thin wrappers over c10 APIs
// from c10/core/ScalarType.h (included transitively via torch/torch.h).
//
// Using c10:: helpers instead of hand-written switch statements ensures
// coverage of newer dtypes (UInt16/32/64, Float8 variants, etc.).

// Promote half/bfloat16 to float32 for computation accuracy
inline at::ScalarType to_opmath_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return at::kFloat;
    case at::kBFloat16:
      return at::kFloat;
    case at::kComplexHalf:
      return at::kComplexFloat;
    default:
      return dtype;
  }
}

// Compute (computation_dtype, result_dtype) for a set of inputs given a
// promotion rule.  Mirrors Python's torch._prims_common.elementwise_dtypes.
//
// Optimized: avoids creating temporary at::scalar_tensor objects for scalar
// args — uses ScalarType directly for type promotion.
template <typename TensorContainer>
inline std::pair<at::ScalarType, at::ScalarType> compute_promoted_dtype(
    const TensorContainer& inputs,
    const std::vector<double>& scalar_args,
    const std::vector<bool>& is_tensor_mask,
    const PromotionRule& rule) {
  // Collect ScalarTypes directly — no tensor allocation needed
  at::ScalarType common_dtype = at::ScalarType::Undefined;
  bool first = true;

  for (int idx : rule.arg_indices) {
    at::ScalarType this_dtype;
    if (is_tensor_mask[idx]) {
      int tensor_idx = 0;
      for (int k = 0; k < idx; ++k) {
        if (is_tensor_mask[k]) tensor_idx++;
      }
      this_dtype = inputs[tensor_idx].scalar_type();
    } else {
      // Python scalars default to float64 (double)
      this_dtype = at::kDouble;
    }

    if (first) {
      common_dtype = this_dtype;
      first = false;
    } else {
      common_dtype = at::promote_types(common_dtype, this_dtype);
    }
  }

  at::ScalarType computation_dtype = common_dtype;
  at::ScalarType result_dtype = common_dtype;

  switch (rule.method) {
    case TypePromotionKind::DEFAULT:
      computation_dtype = to_opmath_dtype(common_dtype);
      result_dtype = common_dtype;
      break;

    case TypePromotionKind::NO_OPMATH:
      computation_dtype = common_dtype;
      result_dtype = common_dtype;
      break;

    case TypePromotionKind::INT_TO_FLOAT:
      if (c10::isIntegralType(common_dtype, /*includeBool=*/true)) {
        computation_dtype = at::kFloat;
        result_dtype = at::kFloat;
      } else {
        computation_dtype = to_opmath_dtype(common_dtype);
        result_dtype = common_dtype;
      }
      break;

    case TypePromotionKind::ALWAYS_BOOL:
      computation_dtype = to_opmath_dtype(common_dtype);
      result_dtype = at::kBool;
      break;

    case TypePromotionKind::COMPLEX_TO_FLOAT:
      if (c10::isComplexType(common_dtype)) {
        result_dtype = c10::toRealValueType(common_dtype);
      } else {
        result_dtype = common_dtype;
      }
      computation_dtype = to_opmath_dtype(common_dtype);
      break;

    case TypePromotionKind::BOOL_TO_LONG:
      if (common_dtype == at::kBool) {
        computation_dtype = at::kLong;
        result_dtype = at::kLong;
      } else {
        computation_dtype = to_opmath_dtype(common_dtype);
        result_dtype = common_dtype;
      }
      break;
  }

  return {computation_dtype, result_dtype};
}

// ==========================================================================
// ==========================================================================
// Helper: create a view with overridden shape/strides via as_strided.
//
// Replaces the custom StridedBuffer struct. PyTorch's as_strided creates
// a view of the same storage with the given shape and strides, which is
// exactly what the kernel launch needs.
//   - Fast path: collapse to 1D → as_strided({numel}, {1})
//   - Slow path: use broadcasted strides → as_strided(task_shape, bcast_strides)
// ==========================================================================

inline at::Tensor make_strided_view(const at::Tensor& base,
                                    const c10::SmallVector<int64_t, 8>& shape,
                                    const c10::SmallVector<int64_t, 8>& strides) {
  return base.as_strided(shape, strides, base.storage_offset());
}

// ==========================================================================
// Generic dispatch function with fast-path optimization and
// pre-allocated output support.
//
// Mirrors Python's PointwiseDynamicFunction.prepare_args + __call__
// ==========================================================================

// Internal implementation that takes a pre-resolved kernel registry to avoid
// repeated string-based hash map lookups.  The public dispatch_pointwise()
// overloads resolve the registry once and delegate here.
inline at::Tensor dispatch_pointwise_impl(const std::unordered_map<int, KernelInfo>& op_registry,
                                          const std::vector<at::Tensor>& inputs_orig,
                                          const std::vector<double>& scalar_args,
                                          const std::vector<bool>& is_tensor_mask,
                                          const std::vector<c10::optional<at::Tensor>>& pre_outputs) {
  // =========================================================================
  // Device promotion: move CPU scalar tensors (0-dim) to the target device.
  // =========================================================================
  c10::Device target_device = c10::kCPU;
  for (const auto& t : inputs_orig) {
    if (t.device() != c10::kCPU) {
      target_device = t.device();
      break;
    }
  }
  c10::SmallVector<at::Tensor, 4> inputs;
  inputs.reserve(inputs_orig.size());
  for (const auto& t : inputs_orig) {
    if (t.device() == c10::kCPU && t.dim() == 0 && target_device != c10::kCPU) {
      inputs.push_back(t.to(target_device));
    } else {
      inputs.push_back(t);
    }
  }

  // Lookup kernel metadata (rank 0 entry carries shared metadata)
  auto meta_it = op_registry.find(0);
  if (meta_it == op_registry.end()) {
    throw std::runtime_error("No kernel metadata found (rank 0)");
  }
  const KernelInfo* info_meta = &meta_it->second;
  int num_outputs = info_meta->num_outputs;

  // =========================================================================
  // Collect pre-allocated outputs and determine which need allocation
  // =========================================================================
  c10::SmallVector<at::Tensor, 2> out_tensors;
  c10::SmallVector<int, 2> outputs_that_need_allocation;
  for (int i = 0; i < num_outputs; ++i) {
    if (i < static_cast<int>(pre_outputs.size()) && pre_outputs[i].has_value()) {
      out_tensors.push_back(pre_outputs[i].value());
    } else {
      outputs_that_need_allocation.push_back(i);
    }
  }

  // Compute broadcast shape from input tensors
  c10::SmallVector<c10::SmallVector<int64_t, 8>, 4> shapes;
  shapes.reserve(inputs.size());
  for (const auto& t : inputs) {
    shapes.emplace_back(t.sizes().begin(), t.sizes().end());
  }
  auto out_shape = broadcast_shapes(shapes);

  // =========================================================================
  // Validate pre-allocated outputs
  // =========================================================================
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    const auto& ot = out_tensors[i];
    auto ot_sizes = ot.sizes();
    if (ot_sizes.size() != out_shape.size() ||
        !std::equal(ot_sizes.begin(), ot_sizes.end(), out_shape.begin())) {
      throw std::runtime_error("out tensor at index " + std::to_string(i) +
                               " has invalid shape, expected broadcast shape");
    }
    if (has_internal_overlapping(ot)) {
      throw std::runtime_error("Pointwise output arguments should not have internal overlapping.");
    }
  }

  // =========================================================================
  // Fast-path detection: includes BOTH pre-allocated outputs and inputs
  // =========================================================================
  c10::SmallVector<at::Tensor, 4> all_tensors;
  all_tensors.reserve(out_tensors.size() + inputs.size());
  all_tensors.insert(all_tensors.end(), out_tensors.begin(), out_tensors.end());
  all_tensors.insert(all_tensors.end(), inputs.begin(), inputs.end());

  // =========================================================================
  // INT32_MAX check
  // =========================================================================
  constexpr int64_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max();
  bool prefer_block_pointer = true;
  if (!all_tensors.empty() && all_tensors[0].numel() > INT32_MAX_VAL) {
    prefer_block_pointer = false;
  }

  c10::SmallVector<int64_t, 8> task_shape;
  int ndim;

  bool fast_path = use_fast_path(all_tensors);

  if (fast_path) {
    int64_t numel = 1;
    for (auto d : out_shape) numel *= d;
    task_shape = {numel};
    ndim = 1;
  } else {
    task_shape = out_shape;
    ndim = static_cast<int>(out_shape.size());
  }

  // Lookup kernel by effective rank — direct map lookup, no string hash
  auto rank_it = op_registry.find(ndim);
  if (rank_it == op_registry.end()) {
    throw std::runtime_error("No kernel for rank " + std::to_string(ndim));
  }
  const KernelInfo* info = &rank_it->second;

  // =========================================================================
  // Dtype promotion: only for outputs that need allocation
  // =========================================================================
  const std::vector<bool>& tensor_mask =
      !is_tensor_mask.empty() ? is_tensor_mask : [&]() -> const std::vector<bool>& {
    // Build default mask on first use; cached via static thread_local
    // to avoid repeated allocation for the common case.
    thread_local std::vector<bool> default_mask;
    int total_args = info->num_input_tensors + info->num_non_tensor_inputs;
    default_mask.assign(total_args, false);
    for (int i = 0; i < info->num_input_tensors; ++i) {
      default_mask[i] = true;
    }
    return default_mask;
  }();

  c10::SmallVector<at::ScalarType, 2> alloc_dtypes;
  for (int out_idx : outputs_that_need_allocation) {
    if (out_idx < static_cast<int>(info->promotion_rules.size())) {
      auto [comp_dtype, result_dtype] =
          compute_promoted_dtype(inputs, scalar_args, tensor_mask, info->promotion_rules[out_idx]);
      alloc_dtypes.push_back(result_dtype);
    } else {
      alloc_dtypes.push_back(inputs[0].scalar_type());
    }
  }

  // =========================================================================
  // Allocate missing outputs
  // =========================================================================
  c10::SmallVector<at::Tensor, 2> allocated_outputs;
  if (fast_path) {
    for (auto dtype : alloc_dtypes) {
      allocated_outputs.push_back(at::empty_like(all_tensors[0], at::TensorOptions().dtype(dtype)));
    }
  } else {
    const at::Tensor* template_tensor = nullptr;
    for (const auto& t : all_tensors) {
      auto t_sizes = t.sizes();
      if (t_sizes.size() == task_shape.size() &&
          std::equal(t_sizes.begin(), t_sizes.end(), task_shape.begin())) {
        template_tensor = &t;
        break;
      }
    }
    for (auto dtype : alloc_dtypes) {
      if (template_tensor) {
        allocated_outputs.push_back(at::empty_like(*template_tensor, at::TensorOptions().dtype(dtype)));
      } else {
        allocated_outputs.push_back(at::empty(task_shape, inputs[0].options().dtype(dtype)));
      }
    }
  }

  // =========================================================================
  // Build final output list: merge pre-allocated + newly allocated
  // =========================================================================
  c10::SmallVector<at::Tensor, 2> outputs(num_outputs);
  int alloc_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    if (i < static_cast<int>(pre_outputs.size()) && pre_outputs[i].has_value()) {
      outputs[i] = pre_outputs[i].value();
    } else {
      outputs[i] = allocated_outputs[alloc_idx++];
    }
  }

  // Early return for empty tensors
  if (outputs[0].numel() == 0) {
    return outputs[0];
  }

  // =========================================================================
  // Create strided views via as_strided
  // =========================================================================
  c10::SmallVector<int64_t, 8> unit_strides(ndim, 1);

  c10::SmallVector<at::Tensor, 4> input_views;
  input_views.reserve(inputs.size());
  if (fast_path) {
    for (const auto& t : inputs) {
      input_views.push_back(make_strided_view(t, task_shape, unit_strides));
    }
  } else {
    for (const auto& t : inputs) {
      input_views.push_back(
          make_strided_view(t, task_shape, broadcasted_stride(t.sizes(), t.strides(), out_shape)));
    }
  }

  c10::SmallVector<at::Tensor, 2> output_views;
  output_views.reserve(outputs.size());
  if (fast_path) {
    for (const auto& t : outputs) {
      output_views.push_back(make_strided_view(t, task_shape, unit_strides));
    }
  } else {
    for (const auto& t : outputs) {
      output_views.push_back(
          make_strided_view(t, task_shape, broadcasted_stride(t.sizes(), t.strides(), task_shape)));
    }
  }

  // =========================================================================
  // Kernel lookup and launch params
  // =========================================================================
  TritonJITFunction& kernel = TritonJITFunction::get_instance(info->file_path, info->kernel_name);

  int64_t num_tasks = 1;
  for (auto d : task_shape) num_tasks *= d;

  c10::SmallVector<int64_t, 8> nd_tile_sizes;
  int64_t tile_size;
  int64_t num_tiles;

  bool use_nd_tiles = !info->is_1d_tile && ndim > 1;
  if (use_nd_tiles) {
    nd_tile_sizes = heuristics_for_nd_tile_sizes(info->max_tile_size, task_shape);
    tile_size = 1;
    num_tiles = 1;
    for (int d = 0; d < ndim; ++d) {
      tile_size *= nd_tile_sizes[d];
      num_tiles *= (task_shape[d] + nd_tile_sizes[d] - 1) / nd_tile_sizes[d];
    }
  } else {
    tile_size = heuristics_for_tile_size(num_tasks);
    num_tiles = (num_tasks + tile_size - 1) / tile_size;
  }

  int64_t num_ctas = std::min(num_tiles, int64_t(65535));
  int64_t tiles_per_cta = (num_tiles + num_ctas - 1) / num_ctas;
  int num_warps = heuristics_for_num_warps(tile_size);
  int64_t one_tile_per_cta = (tiles_per_cta == 1) ? 1 : 0;

  // Get stream
  c10::DeviceGuard guard(inputs[0].device());
  c10::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(cuda_stream.stream());

  // =========================================================================
  // Build kernel arguments
  // =========================================================================
  {
    ParameterBuffer buffer;
    const auto& ssig = kernel.get_static_sig();
    buffer.reserve(ssig.num_args);
    c10::SmallVector<std::string> signature;
    signature.reserve(ssig.num_args);
    ArgHandle handler = {ssig, buffer, signature, 0};

    // --- 1. Data args ---
    int tensor_idx = 0;
    int scalar_idx = 0;
    for (size_t i = 0; i < tensor_mask.size(); ++i) {
      if (tensor_mask[i]) {
        handler.handle_arg(input_views[tensor_idx]);
        tensor_idx++;
      } else {
        handler.handle_arg(scalar_args[scalar_idx]);
        scalar_idx++;
      }
    }
    for (int i = 0; i < num_outputs; ++i) {
      handler.handle_arg(output_views[i]);
    }

    // --- 2. Per-input strides + stride_order ---
    bool with_stride_order = info->is_block_pointer;

    if (ndim > 0) {
      for (size_t i = 0; i < input_views.size(); ++i) {
        auto strides = input_views[i].strides();
        for (int d = 0; d < ndim; ++d) {
          handler.handle_arg(strides[d]);
        }
        if (with_stride_order) {
          auto stride_order = (ndim >= 2) ? compute_stride_order(strides) : c10::SmallVector<int64_t, 8> {0};
          for (int d = 0; d < ndim; ++d) {
            handler.handle_arg(stride_order[d]);
          }
        }
      }

      // --- 3. Per-output strides + stride_order ---
      for (size_t i = 0; i < output_views.size(); ++i) {
        auto strides = output_views[i].strides();
        for (int d = 0; d < ndim; ++d) {
          handler.handle_arg(strides[d]);
        }
        if (with_stride_order) {
          auto stride_order = (ndim >= 2) ? compute_stride_order(strides) : c10::SmallVector<int64_t, 8> {0};
          for (int d = 0; d < ndim; ++d) {
            handler.handle_arg(stride_order[d]);
          }
        }
      }

      // --- 4. Shape dims ---
      for (int d = 0; d < ndim; ++d) {
        handler.handle_arg(task_shape[d]);
      }

      // --- 5. Trailing launch parameters ---
      handler.handle_arg(num_tasks);
      handler.handle_arg(tiles_per_cta);

      if (use_nd_tiles) {
        for (int d = 0; d < ndim; ++d) {
          handler.handle_arg(nd_tile_sizes[d]);
        }
      } else {
        handler.handle_arg(tile_size);
      }
      handler.handle_arg(one_tile_per_cta);
    }

    handler.append_global_scratch();
    handler.append_global_scratch();

    std::string full_signature = join_sig(signature);

    c10::SmallVector<void*> ptrs = buffer.get_ptrs();
    kernel.launch_with_raw_args(raw_stream,
                                static_cast<unsigned int>(num_ctas),
                                1,
                                1,
                                num_warps,
                                1 /* num_stages */,
                                full_signature,
                                ptrs.data(),
                                ptrs.size());
  }

  return outputs[0];
}

// ==========================================================================
// Public API: dispatch_pointwise with string-based op_name lookup.
// Resolves the registry once per op via static local cache, then delegates
// to dispatch_pointwise_impl which uses the pre-resolved registry.
// ==========================================================================

inline at::Tensor dispatch_pointwise(const std::string& op_name,
                                     const std::vector<at::Tensor>& inputs_orig,
                                     const std::vector<double>& scalar_args = {},
                                     const std::vector<bool>& is_tensor_mask = {},
                                     const std::vector<c10::optional<at::Tensor>>& pre_outputs = {}) {
  // Resolve op registry — the KERNEL_REGISTRY is a compile-time constant map,
  // so this lookup is safe to cache for the process lifetime.
  auto op_it = KERNEL_REGISTRY.find(op_name);
  if (op_it == KERNEL_REGISTRY.end()) {
    throw std::runtime_error("Unknown op: " + op_name);
  }
  return dispatch_pointwise_impl(op_it->second, inputs_orig, scalar_args, is_tensor_mask, pre_outputs);
}

// Convenience overload: dispatch with a single pre-allocated output tensor
inline at::Tensor dispatch_pointwise_out(const std::string& op_name,
                                         const std::vector<at::Tensor>& inputs,
                                         at::Tensor& out,
                                         const std::vector<double>& scalar_args = {},
                                         const std::vector<bool>& is_tensor_mask = {}) {
  std::vector<c10::optional<at::Tensor>> pre_outputs = {out};
  return dispatch_pointwise(op_name, inputs, scalar_args, is_tensor_mask, pre_outputs);
}

}  // namespace pointwise_dynamic
