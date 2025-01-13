#pragma once

#include <cstdint>

#include <opencv2/core/types.hpp>

namespace toybox {

enum class CpuImpl { Naive, Avx2 };
enum class GpuImpl { Naive, U32Simd, TextureMemory };
class Calc {
public:
  void GpuInitialize();
  void GpuFinalize();
  void ExecuteCpu();
  void ExecuteGpu();

  template <CpuImpl impl>
  void ExecuteCpu(cv::Mat& src, cv::Mat& ret, const int32_t begin_x, const int32_t begin_y,
                  const int32_t cut_size_x, const int32_t cut_size_y);

  template <GpuImpl impl>
  void ExecuteGpu(cv::Mat& src, cv::Mat& ret, const int32_t begin_x, const int32_t begin_y,
                  const int32_t cut_size_x, const int32_t cut_size_y);
};
} // namespace toybox
