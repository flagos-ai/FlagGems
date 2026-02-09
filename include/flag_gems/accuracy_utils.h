#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

namespace flag_gems::accuracy_utils {

extern bool TO_CPU;

struct CheckCloseResult {
  bool ok;
  std::string message;
};

torch::Tensor to_reference(torch::Tensor inp, bool upcast = false);

torch::Tensor to_cpu(torch::Tensor res, const torch::Tensor& ref);

CheckCloseResult gems_assert_close(torch::Tensor res,
                                   torch::Tensor ref,
                                   c10::ScalarType dtype = c10::ScalarType::Undefined,
                                   bool equal_nan = false,
                                   int64_t reduce_dim = 1,
                                   float atol = 1e-4);

CheckCloseResult gems_assert_equal(torch::Tensor res, torch::Tensor ref, bool equal_nan = false);

// Temporary: relax precision for Triton div (no pointwise_dynamic support).
// Will remove once implementation supports pointwise_dynamic.
CheckCloseResult gems_assert_close_div_factor(torch::Tensor res,
                                              torch::Tensor ref,
                                              c10::ScalarType dtype = c10::ScalarType::Undefined,
                                              bool equal_nan = false,
                                              int64_t reduce_dim = 1,
                                              float atol = 1e-4,
                                              bool inplace = false);

}  // namespace flag_gems::accuracy_utils
