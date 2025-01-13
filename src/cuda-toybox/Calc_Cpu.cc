#include "Calc.h"

#include <opencv2/core/core.hpp>

namespace toybox {
template <>
void Calc::ExecuteCpu<CpuImpl::Naive>(cv::Mat& src, cv::Mat& ret, const int32_t begin_x,
                                      const int32_t begin_y, const int32_t cut_size_x,
                                      const int32_t cut_size_y) {
  const int32_t end_x = src.cols - begin_x - cut_size_x;
  const int32_t end_y = src.rows - begin_y - cut_size_y;

  ret = cv::Mat::zeros(src.size(), CV_8UC1);

#pragma omp parallel for
  for (int32_t y = begin_y; y < end_y; y++) {
    for (int32_t x = begin_x; x < end_x; x++) {
      uint8_t mask = std::numeric_limits<uint8_t>::max();
      for (int32_t scan_y = 0; scan_y < cut_size_y; scan_y++) {
        const int32_t pixel_y = y + scan_y;
        uint8_t* sptry        = src.ptr<uint8_t>(pixel_y);
        for (int32_t scan_x = 0; scan_x < cut_size_x; scan_x++) {
          const int32_t pixel_x = x + scan_x;
          mask &= sptry[pixel_x];
        }
      }
      ret.ptr<uint8_t>(y)[x] = mask;
    }
  }
}

template <>
void Calc::ExecuteCpu<CpuImpl::Avx2>(cv::Mat& src, cv::Mat& ret, const int32_t begin_x,
                                     const int32_t begin_y, const int32_t cut_size_x,
                                     const int32_t cut_size_y) {
  const int32_t end_x = src.cols - begin_x - cut_size_x;
  const int32_t end_y = src.rows - begin_y - cut_size_y;

  ret         = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::Mat roi = cv::Mat(ret, cv::Rect(begin_x, begin_y, end_x, end_y));
  roi.setTo(0xFF);

#pragma omp parallel for
  for (int32_t y = begin_y; y < end_y; y++) {
    constexpr int32_t simd_step = 256 / 8 / sizeof(uint8_t);
    for (int32_t x = begin_x; x < end_x; x += simd_step) {
      __m256i mask_v = _mm256_loadu_si256(reinterpret_cast<__m256i*>(ret.ptr<uint8_t>(y) + x));
      for (int32_t scan_y = 0; scan_y < cut_size_y; scan_y++) {
        const int32_t pixel_y = y + scan_y;
        uint8_t* sptry        = src.ptr<uint8_t>(pixel_y);
        for (int32_t scan_x = 0; scan_x < cut_size_x; scan_x++) {
          const int32_t pixel_x = x + scan_x;
          mask_v                = _mm256_and_si256(
              mask_v, _mm256_loadu_si256(reinterpret_cast<__m256i*>(sptry + pixel_x)));
        }
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ret.ptr<uint8_t>(y) + x), mask_v);
    }
  }
}
} // namespace toybox