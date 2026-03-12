// ==============================================================================
// test_pointwise_dynamic.cpp
//
// C++ test for pointwise_dynamic runtime — add operator only
// ==============================================================================

#include <cassert>
#include <cmath>
#include <iostream>

#include "pointwise_runtime.h"
#include "torch/torch.h"

namespace {

constexpr float RTOL = 1e-4f;
constexpr float ATOL = 1e-5f;

bool tensors_close(const at::Tensor& a, const at::Tensor& b) {
  if (a.sizes() != b.sizes()) {
    std::cerr << "Shape mismatch: " << a.sizes() << " vs " << b.sizes() << std::endl;
    return false;
  }
  auto diff = (a.to(torch::kFloat) - b.to(torch::kFloat)).abs();
  auto max_diff = diff.max().item<float>();
  auto threshold = ATOL + RTOL * b.to(torch::kFloat).abs().max().item<float>();
  if (max_diff > threshold) {
    std::cerr << "Value mismatch: max_diff=" << max_diff << ", threshold=" << threshold << std::endl;
    return false;
  }
  return true;
}

// ==============================================================================
// Test cases — add_func only
// ==============================================================================

bool test_add_same_shape() {
  std::cout << "  test_add_same_shape... ";
  auto a = torch::randn({3, 4}, torch::kCUDA);
  auto b = torch::randn({3, 4}, torch::kCUDA);
  auto triton_result = pointwise_dynamic::add_func(a, b);
  auto torch_result = a + b;
  if (tensors_close(triton_result, torch_result)) {
    std::cout << "PASSED" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_add_broadcast() {
  std::cout << "  test_add_broadcast... ";
  auto a = torch::randn({3, 1}, torch::kCUDA);
  auto b = torch::randn({1, 4}, torch::kCUDA);
  auto triton_result = pointwise_dynamic::add_func(a, b);
  auto torch_result = a + b;
  if (triton_result.sizes() != torch_result.sizes()) {
    std::cout << "FAILED (shape mismatch)" << std::endl;
    return false;
  }
  if (tensors_close(triton_result, torch_result)) {
    std::cout << "PASSED (broadcast to " << torch_result.sizes() << ")" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_add_with_alpha() {
  std::cout << "  test_add_with_alpha... ";
  auto a = torch::randn({3, 4}, torch::kCUDA);
  auto b = torch::randn({3, 4}, torch::kCUDA);
  double alpha = 2.5;
  auto triton_result = pointwise_dynamic::add_func(a, b, alpha);
  auto torch_result = a + alpha * b;
  if (tensors_close(triton_result, torch_result)) {
    std::cout << "PASSED" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_add_different_ranks() {
  std::cout << "  test_add_different_ranks... ";
  bool all_passed = true;

  // Rank 1
  {
    auto a = torch::randn({100}, torch::kCUDA);
    auto b = torch::randn({100}, torch::kCUDA);
    if (!tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
      std::cerr << "Rank 1 failed" << std::endl;
      all_passed = false;
    }
  }
  // Rank 2
  {
    auto a = torch::randn({10, 10}, torch::kCUDA);
    auto b = torch::randn({10, 10}, torch::kCUDA);
    if (!tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
      std::cerr << "Rank 2 failed" << std::endl;
      all_passed = false;
    }
  }
  // Rank 3
  {
    auto a = torch::randn({4, 5, 6}, torch::kCUDA);
    auto b = torch::randn({4, 5, 6}, torch::kCUDA);
    if (!tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
      std::cerr << "Rank 3 failed" << std::endl;
      all_passed = false;
    }
  }
  // Rank 4
  {
    auto a = torch::randn({2, 3, 4, 5}, torch::kCUDA);
    auto b = torch::randn({2, 3, 4, 5}, torch::kCUDA);
    if (!tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
      std::cerr << "Rank 4 failed" << std::endl;
      all_passed = false;
    }
  }

  std::cout << (all_passed ? "PASSED" : "FAILED") << std::endl;
  return all_passed;
}

bool test_add_large_tensor() {
  std::cout << "  test_add_large_tensor... ";
  auto a = torch::randn({1024, 1024}, torch::kCUDA);
  auto b = torch::randn({1024, 1024}, torch::kCUDA);
  if (tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
    std::cout << "PASSED" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_add_empty_tensor() {
  std::cout << "  test_add_empty_tensor... ";
  auto a = torch::randn({0, 4}, torch::kCUDA);
  auto b = torch::randn({0, 4}, torch::kCUDA);
  auto result = pointwise_dynamic::add_func(a, b);
  if (result.numel() == 0) {
    std::cout << "PASSED" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_add_fast_path_contiguous() {
  std::cout << "  test_add_fast_path_contiguous... ";
  auto a = torch::randn({2, 3, 4, 5}, torch::kCUDA);
  auto b = torch::randn({2, 3, 4, 5}, torch::kCUDA);
  if (tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
    std::cout << "PASSED" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_add_non_contiguous() {
  std::cout << "  test_add_non_contiguous... ";
  auto a = torch::randn({4, 5}, torch::kCUDA).t();  // 5x4, non-contiguous
  auto b = torch::randn({4, 5}, torch::kCUDA).t();
  if (tensors_close(pointwise_dynamic::add_func(a, b), a + b)) {
    std::cout << "PASSED" << std::endl;
    return true;
  }
  std::cout << "FAILED" << std::endl;
  return false;
}

bool test_fast_path_helpers() {
  std::cout << "  test_fast_path_helpers... ";
  auto a = torch::randn({3, 4}, torch::kCUDA);
  auto b = torch::randn({3, 4}, torch::kCUDA);
  auto c = torch::randn({3, 5}, torch::kCUDA);

  std::vector<at::Tensor> same_shape = {a, b};
  std::vector<at::Tensor> diff_shape = {a, c};

  bool ok = pointwise_dynamic::all_same_shape(same_shape) && !pointwise_dynamic::all_same_shape(diff_shape) &&
            pointwise_dynamic::all_contiguous(same_shape) && pointwise_dynamic::use_fast_path(same_shape) &&
            !pointwise_dynamic::use_fast_path(diff_shape);

  std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
  return ok;
}

}  // namespace

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "pointwise_dynamic C++ Tests (add only)" << std::endl;
  std::cout << "========================================" << std::endl;

  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA not available, skipping tests" << std::endl;
    return 0;
  }
  std::cout << "CUDA available: " << torch::cuda::device_count() << " device(s)" << std::endl;
  std::cout << std::endl;

  int passed = 0, failed = 0;
  auto run_test = [&](bool (*test_fn)(), const char* name) {
    try {
      if (test_fn())
        passed++;
      else
        failed++;
    } catch (const std::exception& e) {
      std::cerr << "  " << name << "... EXCEPTION: " << e.what() << std::endl;
      failed++;
    }
    // Synchronize after each test so async CUDA errors surface here
    // rather than corrupting the next test.
    torch::cuda::synchronize();
  };

  std::cout << "add_func tests:" << std::endl;
  run_test(test_add_same_shape, "test_add_same_shape");
  run_test(test_add_broadcast, "test_add_broadcast");
  run_test(test_add_with_alpha, "test_add_with_alpha");
  run_test(test_add_different_ranks, "test_add_different_ranks");
  run_test(test_add_large_tensor, "test_add_large_tensor");
  run_test(test_add_empty_tensor, "test_add_empty_tensor");

  std::cout << "\nFast-path tests:" << std::endl;
  run_test(test_add_fast_path_contiguous, "test_add_fast_path_contiguous");
  run_test(test_add_non_contiguous, "test_add_non_contiguous");
  run_test(test_fast_path_helpers, "test_fast_path_helpers");

  std::cout << "\n========================================" << std::endl;
  std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  return (failed == 0) ? 0 : 1;
}
