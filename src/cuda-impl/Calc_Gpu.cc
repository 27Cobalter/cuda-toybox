#include "../cuda-toybox/Calc.h"

#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "KernelImpl.cuh"

#pragma comment( \
    lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64/cudart_static.lib")

namespace toybox {
__host__ void Calc::GpuInitialize() {
  cudaSetDevice(0);
}

__host__ void Calc::GpuFinalize() {
  cudaDeviceReset();
}

template void Calc::ExecuteGpu<GpuImpl::Naive>(cv::Mat& src, cv::Mat& ret,
                                               const int32_t begin_x, const int32_t begin_y,
                                               const int32_t cut_size_x,
                                               const int32_t cut_size_y);
template void Calc::ExecuteGpu<GpuImpl::U32Simd>(cv::Mat& src, cv::Mat& ret,
                                                 const int32_t begin_x, const int32_t begin_y,
                                                 const int32_t cut_size_x,
                                                 const int32_t cut_size_y);

template <GpuImpl impl>
__host__ void Calc::ExecuteGpu(cv::Mat& src, cv::Mat& ret, const int32_t begin_x,
                               const int32_t begin_y, const int32_t cut_size_x,
                               const int32_t cut_size_y) {
  const int32_t end_x = src.cols - begin_x - cut_size_x;
  const int32_t end_y = src.rows - begin_y - cut_size_y;

  const int32_t elem_size = src.size().area();
  thrust::device_vector<uint8_t> src_dev(elem_size);
  thrust::device_vector<uint8_t> ret_dev(elem_size);
  thrust::copy(src.ptr<uint8_t>(), src.ptr<uint8_t>() + elem_size, src_dev.begin());

  if constexpr (impl == GpuImpl::Naive) {
    dim3 block(512, 2);
    int32_t grid_x = std::ceil(static_cast<float>(end_x - begin_x) / block.x);
    int32_t grid_y = std::ceil(static_cast<float>(end_y - begin_y) / block.y);
    dim3 grid(grid_x, grid_y);
    gpu::Kernel_Naive<<<grid, block>>>(begin_x, begin_y, end_x, end_y, src_dev.data().get(),
                                       ret_dev.data().get(), src.cols, cut_size_x, cut_size_y);
  } else if constexpr (impl == GpuImpl::U32Simd) {
    dim3 block(512, 2);
    int32_t grid_x = std::ceil(static_cast<float>(end_x - begin_x) / 4 / block.x);
    int32_t grid_y = std::ceil(static_cast<float>(end_y - begin_y) / block.y);
    dim3 grid(grid_x, grid_y);
    gpu::Kernel_U32Simd<<<grid, block>>>(begin_x, begin_y, end_x, end_y, src_dev.data().get(),
                                         ret_dev.data().get(), src.cols, cut_size_x,
                                         cut_size_y);
  }

  auto status = cudaGetLastError();
  assert(status == cudaSuccess);

  cudaDeviceSynchronize();

  thrust::copy(ret_dev.begin(), ret_dev.end(), ret.ptr<uint8_t>());
  assert(status == cudaSuccess);

  return;
}

template <>
__host__ void Calc::ExecuteGpu<GpuImpl::TextureMemory>(cv::Mat& src, cv::Mat& ret,
                                                       const int32_t begin_x,
                                                       const int32_t begin_y,
                                                       const int32_t cut_size_x,
                                                       const int32_t cut_size_y) {
  const int32_t end_x = src.cols - begin_x - cut_size_x;
  const int32_t end_y = src.rows - begin_y - cut_size_y;

  const int32_t elem_size = src.size().area();
  thrust::device_vector<uint8_t> src_dev(elem_size);
  thrust::device_vector<uint8_t> ret_dev(elem_size);
  thrust::copy(src.ptr<uint8_t>(), src.ptr<uint8_t>() + elem_size, src_dev.begin());

  cudaTextureObject_t tex_obj;
  cudaResourceDesc rc_desc;
  cudaTextureDesc tex_desc;
  std::memset(&rc_desc, 0, sizeof(cudaResourceDesc));
  std::memset(&tex_desc, 0, sizeof(cudaTextureDesc));
  rc_desc.resType                  = cudaResourceTypePitch2D;
  rc_desc.res.pitch2D.devPtr       = src_dev.data().get();
  rc_desc.res.pitch2D.desc         = cudaCreateChannelDesc<uint8_t>();
  rc_desc.res.pitch2D.width        = src.cols;
  rc_desc.res.pitch2D.height       = src.rows;
  rc_desc.res.pitch2D.pitchInBytes = src.cols * sizeof(uint8_t);
  tex_desc.normalizedCoords        = false;
  tex_desc.filterMode              = cudaFilterModePoint;
  tex_desc.addressMode[0]          = cudaAddressModeWrap;
  tex_desc.addressMode[1]          = cudaAddressModeWrap;
  tex_desc.readMode                = cudaReadModeElementType;

  cudaCreateTextureObject(&tex_obj, &rc_desc, &tex_desc, nullptr);

  dim3 block(32, 32);
  int32_t grid_x = std::ceil(static_cast<float>(end_x - begin_x) / 4 / block.x);
  int32_t grid_y = std::ceil(static_cast<float>(end_y - begin_y) / block.y);
  dim3 grid(grid_x, grid_y);
  gpu::Kernel_TextureMemory<<<grid, block>>>(begin_x, begin_y, end_x, end_y, tex_obj,
                                             ret_dev.data().get(), src.cols, cut_size_x,
                                             cut_size_y);

  auto status = cudaGetLastError();
  assert(status == cudaSuccess);

  cudaDeviceSynchronize();

  thrust::copy(ret_dev.begin(), ret_dev.end(), ret.ptr<uint8_t>());
  assert(status == cudaSuccess);

  cudaDestroyTextureObject(tex_obj);

  return;
}
} // namespace toybox
