#pragma once

#include "flag_gems/backend_utils.h"

namespace flag_gems::test {

// Convenience aliases — delegate to backend_utils.h
inline torch::Device default_device(int index = 0) {
  return flag_gems::backend::getDefaultDevice(index);
}

inline bool is_device_available() {
  return flag_gems::backend::isDeviceAvailable();
}

inline void synchronize() {
  flag_gems::backend::synchronize();
}

}  // namespace flag_gems::test
