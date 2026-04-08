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

inline std::vector<int64_t> broadcast_shapes(const std::vector<std::vector<int64_t>>& shapes) {
  if (shapes.empty()) return {};
  size_t max_ndim = 0;
  for (const auto& shape : shapes) {
    max_ndim = std::max(max_ndim, shape.size());
  }
  std::vector<int64_t> result(max_ndim, 1);
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

inline std::vector<int64_t> broadcasted_stride(const std::vector<int64_t>& shape,
                                               const std::vector<int64_t>& stride,
                                               const std::vector<int64_t>& target_shape) {
  size_t ndim = target_shape.size();
  size_t offset = ndim - shape.size();
  std::vector<int64_t> result(ndim, 0);
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

inline std::vector<int64_t> compute_stride_order(const std::vector<int64_t>& strides) {
  std::vector<int64_t> order(strides.size());
  std::iota(order.begin(), order.end(), int64_t {0});
  std::sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
    return std::llabs(strides[lhs]) < std::llabs(strides[rhs]);
  });
  return order;
}

// ==========================================================================
// Launch heuristics
// ==========================================================================

inline int64_t heuristics_for_tile_size(int64_t num_tasks) {
  int64_t tile = 1;
  while (tile < 1024 && tile < num_tasks) tile *= 2;
  return std::min(tile, int64_t(1024));
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

inline bool all_same_shape(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) return true;
  const auto& first_sizes = tensors[0].sizes();
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[i].sizes() != first_sizes) {
      return false;
    }
  }
  return true;
}

inline bool all_contiguous(const std::vector<at::Tensor>& tensors) {
  for (const auto& t : tensors) {
    if (!t.is_contiguous()) {
      return false;
    }
  }
  return true;
}

inline bool all_same_stride(const std::vector<at::Tensor>& tensors) {
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
inline bool use_fast_path(const std::vector<at::Tensor>& tensors) {
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
  return !t.is_non_overlapping_and_dense();
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
inline std::pair<at::ScalarType, at::ScalarType> compute_promoted_dtype(
    const std::vector<at::Tensor>& inputs,
    const std::vector<double>& scalar_args,
    const std::vector<bool>& is_tensor_mask,
    const PromotionRule& rule) {
  // Gather the referenced args as tensors for promote_types
  std::vector<at::Tensor> promote_tensors;
  for (int idx : rule.arg_indices) {
    if (is_tensor_mask[idx]) {
      int tensor_idx = 0;
      for (int k = 0; k < idx; ++k) {
        if (is_tensor_mask[k]) tensor_idx++;
      }
      promote_tensors.push_back(inputs[tensor_idx]);
    } else {
      int scalar_idx = 0;
      for (int k = 0; k < idx; ++k) {
        if (!is_tensor_mask[k]) scalar_idx++;
      }
      promote_tensors.push_back(at::scalar_tensor(scalar_args[scalar_idx]));
    }
  }

  // Use PyTorch's promote_types to find the common dtype
  at::ScalarType common_dtype = promote_tensors[0].scalar_type();
  for (size_t i = 1; i < promote_tensors.size(); ++i) {
    common_dtype = at::promote_types(common_dtype, promote_tensors[i].scalar_type());
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
                                    const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& strides) {
  return base.as_strided(shape, strides, base.storage_offset());
}

// ==========================================================================
// Generic dispatch function with fast-path optimization and
// pre-allocated output support.
//
// Mirrors Python's PointwiseDynamicFunction.prepare_args + __call__
// ==========================================================================

inline at::Tensor dispatch_pointwise(const std::string& op_name,
                                     const std::vector<at::Tensor>& inputs_orig,
                                     const std::vector<double>& scalar_args = {},
                                     const std::vector<bool>& is_tensor_mask = {},
                                     const std::vector<c10::optional<at::Tensor>>& pre_outputs = {}) {
  // =========================================================================
  // Device promotion: move CPU scalar tensors (0-dim) to the target device.
  // This handles cases like `cuda_tensor + 1` where PyTorch wraps the Python
  // int as a CPU tensor before dispatching aten::add.Tensor.
  // =========================================================================
  c10::Device target_device = c10::kCPU;
  for (const auto& t : inputs_orig) {
    if (t.device() != c10::kCPU) {
      target_device = t.device();
      break;
    }
  }
  std::vector<at::Tensor> inputs;
  inputs.reserve(inputs_orig.size());
  for (const auto& t : inputs_orig) {
    if (t.device() == c10::kCPU && t.dim() == 0 && target_device != c10::kCPU) {
      inputs.push_back(t.to(target_device));
    } else {
      inputs.push_back(t);
    }
  }

  // Lookup kernel metadata (all ranks share the same metadata)
  const KernelInfo* info_meta = get_kernel_info(op_name, 0);
  if (!info_meta) {
    throw std::runtime_error("Unknown op: " + op_name);
  }
  int num_outputs = info_meta->num_outputs;

  // =========================================================================
  // Collect pre-allocated outputs and determine which need allocation
  // =========================================================================
  std::vector<at::Tensor> out_tensors;
  std::vector<int> outputs_that_need_allocation;
  for (int i = 0; i < num_outputs; ++i) {
    if (i < static_cast<int>(pre_outputs.size()) && pre_outputs[i].has_value()) {
      out_tensors.push_back(pre_outputs[i].value());
    } else {
      outputs_that_need_allocation.push_back(i);
    }
  }

  // Compute broadcast shape from input tensors
  std::vector<std::vector<int64_t>> shapes;
  for (const auto& t : inputs) {
    shapes.push_back(t.sizes().vec());
  }
  auto out_shape = broadcast_shapes(shapes);

  // =========================================================================
  // Validate pre-allocated outputs
  // =========================================================================
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    const auto& ot = out_tensors[i];
    if (ot.sizes().vec() != out_shape) {
      throw std::runtime_error("out tensor at index " + std::to_string(i) +
                               " has invalid shape, expected broadcast shape");
    }
    if (has_internal_overlapping(ot)) {
      throw std::runtime_error("Pointwise output arguments should not have internal overlapping.");
    }
  }

  // =========================================================================
  // Fast-path detection: includes BOTH pre-allocated outputs and inputs
  // Mirrors Python: tensors = out_tensors + in_tensors
  // =========================================================================
  std::vector<at::Tensor> all_tensors;
  all_tensors.reserve(out_tensors.size() + inputs.size());
  all_tensors.insert(all_tensors.end(), out_tensors.begin(), out_tensors.end());
  all_tensors.insert(all_tensors.end(), inputs.begin(), inputs.end());

  // =========================================================================
  // INT32_MAX check: disable block pointer for very large tensors
  // Mirrors Python: if tensors[0].numel() > INT32_MAX:
  //                      self.config.prefer_block_pointer = False
  // =========================================================================
  constexpr int64_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max();
  bool prefer_block_pointer = true;
  if (!all_tensors.empty() && all_tensors[0].numel() > INT32_MAX_VAL) {
    prefer_block_pointer = false;
  }

  std::vector<int64_t> task_shape;
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

  // Lookup kernel by effective rank
  const KernelInfo* info = get_kernel_info(op_name, ndim);
  if (!info) {
    throw std::runtime_error("No kernel for " + op_name + " rank " + std::to_string(ndim));
  }

  // =========================================================================
  // Dtype promotion: only for outputs that need allocation
  // =========================================================================
  std::vector<bool> tensor_mask = is_tensor_mask;
  if (tensor_mask.empty()) {
    int total_args = info->num_input_tensors + info->num_non_tensor_inputs;
    tensor_mask.resize(total_args, false);
    for (int i = 0; i < total_args; ++i) {
      tensor_mask[i] = (i < info->num_input_tensors);
    }
  }

  std::vector<at::ScalarType> alloc_dtypes;
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
  std::vector<at::Tensor> allocated_outputs;
  if (fast_path) {
    for (auto dtype : alloc_dtypes) {
      allocated_outputs.push_back(at::empty_like(all_tensors[0], at::TensorOptions().dtype(dtype)));
    }
  } else {
    const at::Tensor* template_tensor = nullptr;
    for (const auto& t : all_tensors) {
      if (t.sizes().vec() == task_shape) {
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
  std::vector<at::Tensor> outputs(num_outputs);
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
    return outputs[0];  // TODO: multi-output support
  }

  // =========================================================================
  // Create strided views via as_strided
  // Mirrors Python: StridedBuffer(tensor, task_shape, strides)
  //   Fast path: shape={numel}, strides={1}
  //   Slow path: shape=task_shape, strides=broadcasted_stride(...)
  // =========================================================================
  std::vector<int64_t> unit_strides(ndim, 1);  // used in fast path

  // Input views
  std::vector<at::Tensor> input_views;
  input_views.reserve(inputs.size());
  if (fast_path) {
    for (const auto& t : inputs) {
      input_views.push_back(make_strided_view(t, task_shape, unit_strides));
    }
  } else {
    for (const auto& t : inputs) {
      input_views.push_back(
          make_strided_view(t,
                            task_shape,
                            broadcasted_stride(t.sizes().vec(), t.strides().vec(), out_shape)));
    }
  }

  // Output views
  std::vector<at::Tensor> output_views;
  output_views.reserve(outputs.size());
  if (fast_path) {
    for (const auto& t : outputs) {
      output_views.push_back(make_strided_view(t, task_shape, unit_strides));
    }
  } else {
    for (const auto& t : outputs) {
      output_views.push_back(
          make_strided_view(t,
                            task_shape,
                            broadcasted_stride(t.sizes().vec(), t.strides().vec(), task_shape)));
    }
  }

  // =========================================================================
  // Kernel lookup and launch params
  // =========================================================================
  TritonJITFunction& kernel = TritonJITFunction::get_instance(info->file_path, info->kernel_name);

  int64_t num_tasks = 1;
  for (auto d : task_shape) num_tasks *= d;

  int64_t tile_size = heuristics_for_tile_size(num_tasks);
  int64_t num_tiles = (num_tasks + tile_size - 1) / tile_size;
  int64_t num_ctas = std::min(num_tiles, int64_t(65535));
  int64_t tiles_per_cta = (num_tiles + num_ctas - 1) / num_ctas;
  int num_warps = heuristics_for_num_warps(tile_size);
  int64_t one_tile_per_cta = (tiles_per_cta == 1) ? 1 : 0;

  // Get stream
  c10::DeviceGuard guard(inputs[0].device());
  c10::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(cuda_stream.stream());

  // =========================================================================
  // Build kernel arguments dynamically using ParameterBuffer + ArgHandle
  //
  // Argument order for _bptr kernels (mirrors Python gen_kernel_launch):
  //   1. Data args: input tensors (interleaved with scalars by position) + output tensors
  //   2. Per-input: strides[0..ndim-1], stride_order[0..ndim-1]
  //   3. Per-output: strides[0..ndim-1], stride_order[0..ndim-1]
  //   4. Shape dims: shape[0..ndim-1]
  //   5. Trailing: num_tasks, tiles_per_cta, tile_size(s), one_tile_per_cta
  // =========================================================================
  {
    ParameterBuffer buffer;
    const auto& ssig = kernel.get_static_sig();
    buffer.reserve(ssig.num_args);
    c10::SmallVector<std::string> signature;
    signature.reserve(ssig.num_args);
    ArgHandle handler = {ssig, buffer, signature, 0};

    // --- 1. Data args: inputs (tensors + scalars interleaved) + outputs ---
    // Reconstruct the interleaved order using is_tensor_mask:
    //   tensor_mask[i] == true  → next input tensor
    //   tensor_mask[i] == false → next scalar arg
    int tensor_idx = 0;
    int scalar_idx = 0;
    for (size_t i = 0; i < tensor_mask.size(); ++i) {
      if (tensor_mask[i]) {
        // Pass the view tensor — ArgHandle::handle_tensor extracts data_ptr
        handler.handle_arg(input_views[tensor_idx]);
        tensor_idx++;
      } else {
        // Pass scalar as double
        handler.handle_arg(scalar_args[scalar_idx]);
        scalar_idx++;
      }
    }
    // Output tensors
    for (int i = 0; i < num_outputs; ++i) {
      handler.handle_arg(output_views[i]);
    }

    // --- 2. Per-input strides + stride_order (for ndim > 0) ---
    // stride_order is ONLY passed for block-pointer kernels
    // (is_block_pointer=true, is_1d_tile=false).
    // 1D-tile and nd-tile-without-bptr kernels do NOT have stride_order params.
    bool with_stride_order = info->is_block_pointer;

    if (ndim > 0) {
      for (size_t i = 0; i < input_views.size(); ++i) {
        auto strides = input_views[i].strides();
        for (int d = 0; d < ndim; ++d) {
          handler.handle_arg(strides[d]);
        }
        if (with_stride_order) {
          auto stride_order =
              (ndim >= 2) ? compute_stride_order({strides.begin(), strides.end()}) : std::vector<int64_t> {0};
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
          auto stride_order =
              (ndim >= 2) ? compute_stride_order({strides.begin(), strides.end()}) : std::vector<int64_t> {0};
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
      handler.handle_arg(num_tasks);      // num_tasks
      handler.handle_arg(tiles_per_cta);  // tiles_per_cta

      // tile_size layout differs between kernel variants:
      //   1d_tile kernel: single "tile_size: tl.constexpr"
      //   nd_tile kernel: per-dim "tile_size{d}: tl.constexpr"
      if (info->is_1d_tile) {
        handler.handle_arg(tile_size);  // single tile_size
      } else {
        if (ndim == 1) {
          handler.handle_arg(tile_size);  // tile_size0
        } else {
          for (int d = 0; d < ndim; ++d) {
            int64_t dim_tile = heuristics_for_tile_size(task_shape[d]);
            handler.handle_arg(dim_tile);
          }
        }
      }
      handler.handle_arg(one_tile_per_cta);  // one_tile_per_cta
    }

    // Append global scratch (required by triton runtime, twice since triton 3.3)
    handler.append_global_scratch();
    handler.append_global_scratch();

    // Build signature and launch
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

  // =========================================================================
  // Return the output tensor
  // The kernel wrote into the output's storage via the strided view,
  // so we return the original output tensor (not the view).
  // =========================================================================
  // TODO: multi-output support — return tuple of tensors
  return outputs[0];
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
