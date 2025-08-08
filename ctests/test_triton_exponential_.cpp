#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "flag_gems/operators.h"
#include "torch/torch.h"
double calculate_ks_statistic(const std::vector<double>& samples, double lambda) {
  size_t n = samples.size();
  std::vector<double> y_samples;
  y_samples.reserve(n);

  // Y = 1 - exp(-lambda * X)
  for (double x : samples) {
    y_samples.push_back(1.0 - std::exp(-lambda * x));
  }

  std::sort(y_samples.begin(), y_samples.end());
  double d_plus = 0.0;
  double d_minus = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double f_emp = (i + 1.0) / n;
    double f_theo = y_samples[i];

    d_plus = std::max(d_plus, f_emp - f_theo);
    d_minus = std::max(d_minus, f_theo - (static_cast<double>(i) / n));
  }

  return std::max(d_plus, d_minus);
}

double approximate_ks_pvalue(double d, size_t n) {
  double x = d * std::sqrt(n);
  double p = 0.0;

  for (int k = 1; k <= 100; ++k) {
    p += std::pow(-1, k - 1) * std::exp(-2 * k * k * x * x);
  }

  return 2 * p;
}

template <typename T>
void RunExponentialTest(torch::ScalarType dtype) {
  std::vector<int64_t> shape = {20, 320, 15};
  double lambda_values[] = {0.01, 0.5, 100.0};
  const double alpha = 0.05;

  for (double lambda : lambda_values) {
    torch::Tensor x = torch::empty(shape, torch::dtype(dtype)).to(torch::kCUDA);

    x = flag_gems::exponential_(x, lambda);

    torch::Tensor cpu_x = x.cpu().contiguous().view(-1);

    ASSERT_EQ(cpu_x.scalar_type(), dtype) << "Unexpected data type after generation. Expected: " << dtype
                                          << ", Actual: " << cpu_x.scalar_type();
    std::vector<double> samples;
    samples.reserve(cpu_x.numel());

    if (dtype == torch::kFloat16) {
      torch::Tensor float_x = cpu_x.to(torch::kFloat32);
      auto data_ptr = float_x.data_ptr<float>();
      for (int64_t i = 0; i < float_x.numel(); ++i) {
        samples.push_back(static_cast<double>(data_ptr[i]));
      }
    } else if (dtype == torch::kFloat32) {
      auto data_ptr = cpu_x.data_ptr<float>();
      for (int64_t i = 0; i < cpu_x.numel(); ++i) {
        samples.push_back(static_cast<double>(data_ptr[i]));
      }
    } else if (dtype == torch::kFloat64) {
      auto data_ptr = cpu_x.data_ptr<double>();
      for (int64_t i = 0; i < cpu_x.numel(); ++i) {
        samples.push_back(data_ptr[i]);
      }
    }

    double d = calculate_ks_statistic(samples, lambda);

    size_t n = samples.size();
    double p_value = approximate_ks_pvalue(d, n);

    EXPECT_GT(p_value, alpha) << "Failed for lambda=" << lambda << ", dtype=" << dtype
                              << ", p_value=" << p_value;
  }
}

TEST(exponential_op_test, exponential_) {
  // LOG(WARNING) << " test torch::kFloat16";
  // RunExponentialTest<torch::Half>(torch::kFloat16);
  LOG(WARNING) << " test torch::kFloat32";
  RunExponentialTest<float>(torch::kFloat32);  // pytest use type of float32 to test
  // LOG(WARNING) << " test torch::kFloat64";
  // RunExponentialTest<double>(torch::kFloat64);
}
