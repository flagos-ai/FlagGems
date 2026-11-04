#include <gtest/gtest.h>
#include "flag_gems_ops.h"

using namespace flag_gems;

TEST(FlagGemsMM, FallbackSmoke) {
  TensorMeta a {}, b {}, out {};
  mm_fallback(a, b, out);
  SUCCEED();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
