#include <ATen/NestedTensorImpl.h>
#include <torch/extension.h>

at::Tensor gems_nested_copy_wrap(const at::Tensor& buffer,
                                 const at::Tensor& nested_sizes,
                                 const at::Tensor& nested_strides,
                                 const at::Tensor& storage_offsets) {
  // 1. Clone the buffer (the _copy semantic: result owns independent storage)
  auto new_buffer = buffer.clone();

  // 2. Construct a strided-layout NestedTensor wrapping the cloned buffer.
  //    Uses the public 4-arg constructor: (buffer, nested_sizes, nested_strides, storage_offsets).
  //    make_tensor forwards args to the constructor directly.
  return at::detail::make_tensor<at::native::NestedTensorImpl>(new_buffer,
                                                               nested_sizes,
                                                               nested_strides,
                                                               storage_offsets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nested_copy_wrap",
        &gems_nested_copy_wrap,
        "Clone buffer + wrap as strided nested tensor (no vendor delegation)");
}
