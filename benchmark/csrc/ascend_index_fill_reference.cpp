#include <torch/extension.h>

#include <limits>
#include <vector>

#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_index_fill.h>

namespace {

void check_aclnn(aclnnStatus status, const char* operation) {
  TORCH_CHECK(status == OK, operation, " failed with status ", status);
}

aclDataType to_acl_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return ACL_FLOAT16;
    case at::kFloat:
      return ACL_FLOAT;
    case at::kBFloat16:
      return ACL_BF16;
    case at::kLong:
      return ACL_INT64;
    default:
      TORCH_CHECK(false, "Unsupported dtype for ACLNN index_fill reference: ", dtype);
  }
}

class AclTensorHandle {
 public:
  explicit AclTensorHandle(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_contiguous(), "ACLNN reference requires contiguous tensors");

    sizes_.assign(tensor.sizes().begin(), tensor.sizes().end());
    strides_.assign(tensor.strides().begin(), tensor.strides().end());
    tensor_ = aclCreateTensor(
        sizes_.data(),
        sizes_.size(),
        to_acl_dtype(tensor.scalar_type()),
        strides_.data(),
        0,
        ACL_FORMAT_ND,
        sizes_.data(),
        sizes_.size(),
        tensor.data_ptr());
    TORCH_CHECK(tensor_ != nullptr, "aclCreateTensor failed");
  }

  ~AclTensorHandle() {
    if (tensor_ != nullptr) {
      aclDestroyTensor(tensor_);
    }
  }

  AclTensorHandle(const AclTensorHandle&) = delete;
  AclTensorHandle& operator=(const AclTensorHandle&) = delete;

  aclTensor* get() const {
    return tensor_;
  }

 private:
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  aclTensor* tensor_ = nullptr;
};

class AclScalarHandle {
 public:
  AclScalarHandle(void* value, aclDataType dtype) {
    scalar_ = aclCreateScalar(value, dtype);
    TORCH_CHECK(scalar_ != nullptr, "aclCreateScalar failed");
  }

  ~AclScalarHandle() {
    if (scalar_ != nullptr) {
      aclDestroyScalar(scalar_);
    }
  }

  AclScalarHandle(const AclScalarHandle&) = delete;
  AclScalarHandle& operator=(const AclScalarHandle&) = delete;

  aclScalar* get() const {
    return scalar_;
  }

 private:
  aclScalar* scalar_ = nullptr;
};

int64_t normalize_dim(const at::Tensor& input, int64_t dim) {
  TORCH_CHECK(input.dim() > 0, "index_fill expects self to have at least one dimension");
  if (dim < 0) {
    dim += input.dim();
  }
  TORCH_CHECK(dim >= 0 && dim < input.dim(), "index_fill: dim out of range");
  return dim;
}

void validate_inputs(const at::Tensor& input, const at::Tensor& index) {
  TORCH_CHECK(
      input.device().type() == c10::DeviceType::PrivateUse1,
      "ACLNN reference requires an NPU input");
  TORCH_CHECK(index.device() == input.device(), "index and input must be on the same device");
  TORCH_CHECK(index.scalar_type() == at::kLong, "index must have int64 dtype");
  TORCH_CHECK(index.dim() <= 1, "index must be a scalar or one-dimensional tensor");
  TORCH_CHECK(input.is_contiguous(), "ACLNN reference requires contiguous input");
  TORCH_CHECK(index.is_contiguous(), "ACLNN reference requires contiguous index");
}

at::Tensor workspace_tensor(const at::Tensor& input, uint64_t workspace_size) {
  if (workspace_size == 0) {
    return {};
  }
  TORCH_CHECK(
      workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "ACLNN workspace is too large");
  return at::empty(
      {static_cast<int64_t>(workspace_size)}, input.options().dtype(at::kByte));
}

template <typename T>
at::Tensor launch_out_of_place(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    double value) {
  T value_storage = static_cast<T>(value);
  AclTensorHandle input_acl(input);
  AclTensorHandle index_acl(index);
  at::Tensor output = at::empty_like(input);
  AclTensorHandle output_acl(output);
  AclScalarHandle value_acl(&value_storage, to_acl_dtype(input.scalar_type()));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnIndexFillGetWorkspaceSize(
          input_acl.get(),
          dim,
          index_acl.get(),
          value_acl.get(),
          output_acl.get(),
          &workspace_size,
          &executor),
      "aclnnIndexFillGetWorkspaceSize");

  at::Tensor workspace = workspace_tensor(input, workspace_size);
  void* workspace_ptr = workspace.defined() ? workspace.data_ptr() : nullptr;
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(input.get_device()).stream();
  check_aclnn(
      aclnnIndexFill(workspace_ptr, workspace_size, executor, stream),
      "aclnnIndexFill");
  TORCH_CHECK(
      aclrtSynchronizeStream(stream) == ACL_SUCCESS,
      "aclrtSynchronizeStream failed");
  return output;
}

template <typename T>
at::Tensor& launch_in_place(
    at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    double value) {
  T value_storage = static_cast<T>(value);
  AclTensorHandle input_acl(input);
  AclTensorHandle index_acl(index);
  AclScalarHandle value_acl(&value_storage, to_acl_dtype(input.scalar_type()));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  check_aclnn(
      aclnnInplaceIndexFillGetWorkspaceSize(
          input_acl.get(),
          dim,
          index_acl.get(),
          value_acl.get(),
          &workspace_size,
          &executor),
      "aclnnInplaceIndexFillGetWorkspaceSize");

  at::Tensor workspace = workspace_tensor(input, workspace_size);
  void* workspace_ptr = workspace.defined() ? workspace.data_ptr() : nullptr;
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(input.get_device()).stream();
  check_aclnn(
      aclnnInplaceIndexFill(workspace_ptr, workspace_size, executor, stream),
      "aclnnInplaceIndexFill");
  TORCH_CHECK(
      aclrtSynchronizeStream(stream) == ACL_SUCCESS,
      "aclrtSynchronizeStream failed");
  return input;
}

}  // namespace

at::Tensor index_fill(
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    double value) {
  validate_inputs(input, index);
  dim = normalize_dim(input, dim);
  if (input.numel() == 0 || index.numel() == 0) {
    return input.clone();
  }

  c10::DeviceGuard guard(input.device());
  switch (input.scalar_type()) {
    case at::kHalf:
      return launch_out_of_place<c10::Half>(input, dim, index, value);
    case at::kFloat:
      return launch_out_of_place<float>(input, dim, index, value);
    case at::kBFloat16:
      return launch_out_of_place<c10::BFloat16>(input, dim, index, value);
    default:
      TORCH_CHECK(false, "ACLNN index_fill reference supports float16, float32, and bfloat16");
  }
}

at::Tensor index_fill_(
    at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    double value) {
  validate_inputs(input, index);
  dim = normalize_dim(input, dim);
  if (input.numel() == 0 || index.numel() == 0) {
    return input;
  }

  c10::DeviceGuard guard(input.device());
  switch (input.scalar_type()) {
    case at::kHalf:
      return launch_in_place<c10::Half>(input, dim, index, value);
    case at::kFloat:
      return launch_in_place<float>(input, dim, index, value);
    case at::kBFloat16:
      return launch_in_place<c10::BFloat16>(input, dim, index, value);
    default:
      TORCH_CHECK(false, "ACLNN index_fill reference supports float16, float32, and bfloat16");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_fill", &index_fill);
  m.def("index_fill_", &index_fill_);
}
