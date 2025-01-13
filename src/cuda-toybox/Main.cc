#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Calc.h"

int main() {
  constexpr int32_t width  = 8192;
  constexpr int32_t height = 4096;
  cv::Mat src              = cv::Mat::ones(cv::Size(width, height), CV_8UC1) * 255;
  cv::Mat ret              = cv::Mat::zeros(src.size(), src.type());
  cv::Mat ref;

  constexpr int32_t begin_x = 0;
  constexpr int32_t begin_y = 0;
  const int32_t cut_size_x  = (src.cols >> 2) + 1;
  const int32_t cut_size_y  = (src.rows >> 2) + 1;

  src.ptr<uint8_t>(src.rows >> 1)[(src.cols >> 1) - 1]                                = 0;
  src.ptr<uint8_t>((src.rows >> 1) - (src.rows >> 2))[src.cols - (src.cols >> 2) + 1] = 0;
  cv::Mat roi =
      cv::Mat(src, cv::Rect((src.cols >> 1) - (src.cols >> 3), src.rows - (src.rows >> 3),
                            src.cols >> 4, src.rows >> 4));
  roi.setTo(0);

  cv::Size show_size = cv::Size(1024, 1024);
  cv::Mat src_show;
  cv::resize(src, src_show, show_size);

  toybox::Calc calc;

  calc.GpuInitialize();

  std::chrono::steady_clock::time_point start, end;
  std::string impl_name;
  std::vector<std::tuple<std::string, cv::Mat, cv::Mat>> result;

  auto pref = [&](std::string_view name) {
    impl_name = name;
    ret       = cv::Mat::ones(src.size(), src.type()) * 128;
    std::cout << std::format("Exec : {}", name) << std::endl;
  };
  auto suff = [&] {
    std::cout
        << std::format(
               "Duration = {:7.3f}",
               static_cast<float>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) /
                   1000)
        << std::endl;
    std::cout << std::format("PSNR = {}", cv::PSNR(ref, ret)) << std::endl;
    cv::Mat show, diff;
    cv::absdiff(ref, ret, diff);
    cv::resize(diff, diff, show_size);
    cv::resize(ret, show, show_size);
    result.emplace_back(std::make_tuple(impl_name, show, diff));
  };

#define RefId 1
#if RefId == 0
  pref("naive");
  start = std::chrono::high_resolution_clock::now();
  calc.ExecuteCpu<toybox::CpuImpl::Naive>(src, ret, begin_x, begin_y, cut_size_x, cut_size_y);
  end = std::chrono::high_resolution_clock::now();
  ref = ret.clone();
  suff();
#endif

  pref("simd");
  start = std::chrono::high_resolution_clock::now();
  calc.ExecuteCpu<toybox::CpuImpl::Avx2>(src, ret, begin_x, begin_y, cut_size_x, cut_size_y);
#if RefId == 1
  ref = ret.clone();
#endif
  end = std::chrono::high_resolution_clock::now();
  suff();

  pref("gpu_naive");
  start = std::chrono::high_resolution_clock::now();
  calc.ExecuteGpu<toybox::GpuImpl::Naive>(src, ret, begin_x, begin_y, cut_size_x, cut_size_y);
  end = std::chrono::high_resolution_clock::now();
  suff();

  pref("gpu_u32simd");
  start = std::chrono::high_resolution_clock::now();
  calc.ExecuteGpu<toybox::GpuImpl::U32Simd>(src, ret, begin_x, begin_y, cut_size_x, cut_size_y);
  end = std::chrono::high_resolution_clock::now();
  suff();

  pref("gpu_texmem");
  start = std::chrono::high_resolution_clock::now();
  calc.ExecuteGpu<toybox::GpuImpl::TextureMemory>(src, ret, begin_x, begin_y, cut_size_x,
                                                  cut_size_y);
  end = std::chrono::high_resolution_clock::now();
  suff();

  calc.GpuFinalize();

  cv::imshow("src", src_show);
  for (auto [name, show, diff] : result) {
    cv::imshow(name + "_diff", diff);
    cv::imshow(name + "_show", show);
  }
  cv::waitKey();

  return 0;
}