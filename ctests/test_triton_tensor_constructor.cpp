#include <gtest/gtest.h>

#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(zeros_op_test, 2d_tensor) {
  const torch::Device device(torch::kCUDA, 0);
  std::vector<int64_t> shape_0 = {31};
  std::vector<int64_t> shape_1 = {11, 7};
  std::vector<int64_t> shape = {7, 7, 7};
  auto options = torch::TensorOptions().device(device).dtype(torch::kFloat32);

  torch::Tensor ref_empty = torch::empty(shape, options);
  torch::Tensor ref_empty_0 = torch::empty(shape_0, options);
  torch::Tensor ref_empty_1 = torch::empty(shape_1, options);
  ref_empty.fill_(0);
  ref_empty_0.fill_(0);
  ref_empty_1.fill_(0);
  torch::Tensor out_triton = flag_gems::zeros(torch::IntArrayRef(shape),  // size
                                              torch::kFloat32,            // dtype
                                              c10::nullopt,               // layout
                                              device                      // device
  );

  torch::Tensor out_triton_0 = flag_gems::zeros(torch::IntArrayRef(shape_0),  // size
                                                torch::kFloat32,              // dtype
                                                c10::nullopt,                 // layout
                                                device                        // device
  );
  torch::Tensor out_triton_1 = flag_gems::zeros(torch::IntArrayRef(shape_1),  // size
                                                torch::kFloat32,              // dtype
                                                c10::nullopt,                 // layout
                                                device                        // device
  );

  EXPECT_TRUE(torch::all(out_triton == 0).item<bool>());
  EXPECT_TRUE(torch::allclose(out_triton, ref_empty));

  EXPECT_TRUE(torch::all(out_triton_0 == 0).item<bool>());
  EXPECT_TRUE(torch::allclose(out_triton_0, ref_empty_0));

  EXPECT_TRUE(torch::all(out_triton_1 == 0).item<bool>());
  EXPECT_TRUE(torch::allclose(out_triton_1, ref_empty_1));
}
