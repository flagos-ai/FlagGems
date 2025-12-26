// #include <torch/torch.h>
// #include <vector>
// #include "triton_jit/triton_jit_function.h"
// #include "flag_gems/utils.h"

// namespace flag_gems {

// using TritonJITFunction = triton_jit::TritonJITFunction;

// std::vector<int64_t> broadcasted_stride(
//     const std::vector<int64_t>& shape,
//     const std::vector<int64_t>& stride,
//     const std::vector<int64_t>& target_shape) 
// {
//     int ndim_diff = target_shape.size() - shape.size();
//     std::vector<int64_t> full_shape(ndim_diff, 1);
//     full_shape.insert(full_shape.end(), shape.begin(), shape.end());

//     std::vector<int64_t> full_stride(ndim_diff, 0);
//     full_stride.insert(full_stride.end(), stride.begin(), stride.end());

//     std::vector<int64_t> out_stride;
//     for (size_t i = 0; i < target_shape.size(); ++i) {
//         out_stride.push_back(full_shape[i] == 1 ? 0 : full_stride[i]);
//     }
//     return out_stride;
// }

// at::Tensor to_copy(const at::Tensor& x,
//                    c10::optional<at::ScalarType> dtype = c10::nullopt,
//                    c10::optional<c10::Device> device = c10::nullopt,
//                    c10::optional<at::MemoryFormat> memory_format = c10::nullopt) 
// {
//     at::ScalarType target_dtype = dtype.has_value() ? dtype.value() : x.scalar_type();
//     c10::Device target_device = device.has_value() ? device.value() : x.device();
//     at::MemoryFormat target_format = memory_format.has_value() ? memory_format.value() : at::MemoryFormat::Preserve;

//     at::Tensor out = at::empty_like(x, target_dtype, target_device, target_format);

//     int64_t numel = x.numel();
//     if (numel == 0) return out;

//     constexpr int BLOCK_SIZE = 1024;
//     unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

//     if (x.is_contiguous() && out.is_contiguous() && numel <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
//         const TritonJITFunction& kernel = TritonJITFunction::get_instance(
//             (utils::get_triton_src_path() / "copy.py").string(),
//             "copy_kernel_linear"
//         );

//         c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
//         CUstream raw_stream = static_cast<CUstream>(stream.stream());
//         kernel(raw_stream, grid_x, 1, 1, 4, 0, x.view(-1), out.view(-1), numel, BLOCK_SIZE);
//         return out;
//     }

//     std::vector<int64_t> task_shape(x.sizes().begin(), x.sizes().end());
//     int NDIMS = task_shape.size();

//     std::vector<int64_t> src_stride = broadcasted_stride(
//         std::vector<int64_t>(x.sizes().begin(), x.sizes().end()),
//         std::vector<int64_t>(x.strides().begin(), x.strides().end()),
//         task_shape
//     );
//     std::vector<int64_t> dst_stride = broadcasted_stride(
//         std::vector<int64_t>(out.sizes().begin(), out.sizes().end()),
//         std::vector<int64_t>(out.strides().begin(), out.strides().end()),
//         task_shape
//     );

//     const TritonJITFunction& kernel_nd = TritonJITFunction::get_instance(
//         (utils::get_triton_src_path() / "copy.py").string(),
//         "copy_kernel_nd"
//     );

//     c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
//     CUstream raw_stream = static_cast<CUstream>(stream.stream());
//     kernel_nd(raw_stream, grid_x, 1, 1, 4, 0,
//               x, out,
//               torch::from_blob(task_shape.data(), {NDIMS}, torch::kInt64).to(x.device()),
//               torch::from_blob(src_stride.data(), {NDIMS}, torch::kInt64).to(x.device()),
//               torch::from_blob(dst_stride.data(), {NDIMS}, torch::kInt64).to(out.device()),
//               numel,
//               NDIMS,
//               BLOCK_SIZE);
//     return out;
// }


// at::Tensor& copy_(const at::Tensor& src, at::Tensor& dst) {
//     TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have same numel");
//     TORCH_CHECK(src.dim() == dst.dim(), "src and dst must have same dim");

//     int64_t numel = src.numel();
//     if (numel == 0) return dst;

//     constexpr int BLOCK_SIZE = 1024;
//     unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

//     if (src.is_contiguous() && dst.is_contiguous() && numel <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
//         const TritonJITFunction& kernel = TritonJITFunction::get_instance(
//             (utils::get_triton_src_path() / "copy.py").string(),
//             "copy_kernel_linear"
//         );
//         c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
//         CUstream raw_stream = static_cast<CUstream>(stream.stream());
//         kernel(raw_stream, grid_x, 1, 1, 4, 0, src.view(-1), dst.view(-1), numel, BLOCK_SIZE);
//         return dst;
//     }

//     std::vector<int64_t> task_shape(src.sizes().begin(), src.sizes().end());
//     int NDIMS = task_shape.size();

//     std::vector<int64_t> src_stride = broadcasted_stride(
//         std::vector<int64_t>(src.sizes().begin(), src.sizes().end()),
//         std::vector<int64_t>(src.strides().begin(), src.strides().end()),
//         task_shape
//     );
//     std::vector<int64_t> dst_stride = broadcasted_stride(
//         std::vector<int64_t>(dst.sizes().begin(), dst.sizes().end()),
//         std::vector<int64_t>(dst.strides().begin(), dst.strides().end()),
//         task_shape
//     );

//     const TritonJITFunction& kernel_nd = TritonJITFunction::get_instance(
//         (utils::get_triton_src_path() / "copy.py").string(),
//         "copy_kernel_nd"
//     );

//     c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
//     CUstream raw_stream = static_cast<CUstream>(stream.stream());
//     kernel_nd(raw_stream, grid_x, 1, 1, 4, 0,
//               src, dst,
//               torch::from_blob(task_shape.data(), {NDIMS}, torch::kInt64).to(src.device()),
//               torch::from_blob(src_stride.data(), {NDIMS}, torch::kInt64).to(src.device()),
//               torch::from_blob(dst_stride.data(), {NDIMS}, torch::kInt64).to(dst.device()),
//               numel,
//               NDIMS,
//               BLOCK_SIZE);
//     return dst;
// }

// }  // namespace flag_gems



#include "c10/cuda/CUDAStream.h"
#include "c10/util/Logging.h"
#include "torch/extension.h"
#include "triton_jit/triton_jit_function.h"
#include <vector>

namespace flag_gems {

using namespace triton_jit;

at::Tensor to_copy(const at::Tensor& x,
                   c10::optional<at::ScalarType> dtype = c10::nullopt,
                   c10::optional<at::Device> device = c10::nullopt,
                   c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
    TORCH_CHECK(x.layout() == at::Layout::Strided, "Only strided tensors are supported");
    TORCH_CHECK(!x.is_quantized(), "Quantized tensors are not supported");

    auto target_dtype = dtype.has_value() ? dtype.value() : x.scalar_type();
    auto target_device = device.has_value() ? device.value() : x.device();
    auto target_memory_format = memory_format.has_value() ? memory_format.value() : at::MemoryFormat::Preserve;

    if (target_device != x.device()) {
        return x.to(target_device, target_dtype, /*non_blocking=*/false, target_memory_format);
    }

    at::Tensor out = at::empty_like(x, x.options().dtype(target_dtype).device(target_device), target_memory_format);

    const int64_t numel = x.numel();
    if (numel == 0) return out;

    constexpr int BLOCK_SIZE = 1024;
    const unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    c10::DeviceGuard guard(x.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    if (x.is_contiguous() && out.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
        const TritonJITFunction& kernel_linear =
            TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                            "copy_kernel_linear");
        kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, x, out, numel, BLOCK_SIZE);
        return out;
    }

    const int NDIMS = x.dim();
    std::vector<int64_t> task_shape(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> src_stride(x.strides().begin(), x.strides().end());
    std::vector<int64_t> dst_stride(out.strides().begin(), out.strides().end());

    if (NDIMS != task_shape.size()) {
        int ndim_diff = task_shape.size() - NDIMS;
        src_stride.insert(src_stride.begin(), ndim_diff, 0);
        dst_stride.insert(dst_stride.begin(), ndim_diff, 0);
    }
    for (size_t i = 0; i < task_shape.size(); i++) {
        if (task_shape[i] == 1) {
            src_stride[i] = 0;
            dst_stride[i] = 0;
        }
    }

    const TritonJITFunction& kernel_nd =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_nd");
    kernel_nd(raw_stream, grid_x, 1, 1, 4, 0,
              x, out,
              torch::tensor(task_shape, x.device(), torch::kInt64),
              torch::tensor(src_stride, x.device(), torch::kInt64),
              torch::tensor(dst_stride, out.device(), torch::kInt64),
              numel, NDIMS, BLOCK_SIZE);

    return out;
}

at::Tensor& copy_(at::Tensor& dst, const at::Tensor& src) {
    TORCH_CHECK(src.is_tensor(), "src must be a Tensor");
    TORCH_CHECK(dst.numel() == src.numel(), "src and dst must have same number of elements");
    TORCH_CHECK(dst.dim() == src.dim(), "src and dst must have same number of dimensions");
    TORCH_CHECK(!dst._is_zerotensor(), "ZeroTensors are immutable");
    if (src._is_zerotensor()) {
        dst.zero_();
        return dst;
    }

    if (dst.data_ptr() == src.data_ptr()) return dst;

    const int64_t numel = dst.numel();
    if (numel == 0) return dst;

    constexpr int BLOCK_SIZE = 1024;
    const unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    c10::DeviceGuard guard(dst.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());

    if (dst.is_contiguous() && src.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
        const TritonJITFunction& kernel_linear =
            TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                            "copy_kernel_linear");
        kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, src, dst, numel, BLOCK_SIZE);
        return dst;
    }

    const int NDIMS = src.dim();
    std::vector<int64_t> task_shape(src.sizes().begin(), src.sizes().end());
    std::vector<int64_t> src_stride(src.strides().begin(), src.strides().end());
    std::vector<int64_t> dst_stride(dst.strides().begin(), dst.strides().end());

    if (NDIMS != task_shape.size()) {
        int ndim_diff = task_shape.size() - NDIMS;
        src_stride.insert(src_stride.begin(), ndim_diff, 0);
        dst_stride.insert(dst_stride.begin(), ndim_diff, 0);
    }
    for (size_t i = 0; i < task_shape.size(); i++) {
        if (task_shape[i] == 1) {
            src_stride[i] = 0;
            dst_stride[i] = 0;
        }
    }

    const TritonJITFunction& kernel_nd =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_nd");
    kernel_nd(raw_stream, grid_x, 1, 1, 4, 0,
              src, dst,
              torch::tensor(task_shape, src.device(), torch::kInt64),
              torch::tensor(src_stride, src.device(), torch::kInt64),
              torch::tensor(dst_stride, dst.device(), torch::kInt64),
              numel, NDIMS, BLOCK_SIZE);

    return dst;
}

} // namespace flag_gems
