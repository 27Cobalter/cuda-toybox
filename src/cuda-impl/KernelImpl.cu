#pragma once

#include "KernelImpl.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace toybox {
namespace gpu {
__global__ void Kernel_Naive(int32_t begin_x, int32_t begin_y, int32_t end_x, int32_t end_y,
                             uint8_t* src, uint8_t* ret, int32_t width, int32_t cut_size_x,
                             int32_t cut_size_y) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  uint8_t* rety = ret + y * width;

  if (x < begin_x || x >= end_x || y < begin_y || y >= end_y) {
    rety[x] = 0;
    return;
  }

  uint8_t mask = 0xFF;
  for (int32_t scan_y = 0; scan_y < cut_size_y; scan_y++) {
    const int32_t pixel_y = y + scan_y;
    uint8_t* sptry        = src + pixel_y * width;
    for (int32_t scan_x = 0; scan_x < cut_size_x; scan_x++) {
      const int32_t pixel_x = x + scan_x;
      mask &= sptry[pixel_x];
    }
  }
  rety[x] = mask;
}

__global__ void Kernel_U32Simd(int32_t begin_x, int32_t begin_y, int32_t end_x, int32_t end_y,
                               uint8_t* src, uint8_t* ret, int32_t width, int32_t cut_size_x,
                               int32_t cut_size_y) {
  int32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  uint8_t* rety = ret + y * width;

  if (x < begin_x || x >= end_x || y < begin_y || y >= end_y) {
    *reinterpret_cast<uint32_t*>(rety + x) = 0;
    return;
  }

  int32_t shift = (end_x - x) * 8;
  uint8x4_t mask;
  mask.u32 = 0xFFFFFFFF ^ (0xFFFFFFFF << shift);

  for (int32_t scan_y = 0; scan_y < cut_size_y; scan_y++) {
    const int32_t pixel_y = y + scan_y;
    uint8_t* sptry        = src + pixel_y * width;
    for (int32_t scan_x = 0; scan_x < cut_size_x; scan_x++) {
      const int32_t pixel_x = x + scan_x;
      mask.u8[0] &= *reinterpret_cast<uint8_t*>(sptry + pixel_x);
      mask.u8[1] &= *reinterpret_cast<uint8_t*>(sptry + pixel_x + 1);
      mask.u8[2] &= *reinterpret_cast<uint8_t*>(sptry + pixel_x + 2);
      mask.u8[3] &= *reinterpret_cast<uint8_t*>(sptry + pixel_x + 3);
    }
  }
  *reinterpret_cast<uint32_t*>(rety + x) = mask.u32;
}

__global__ void Kernel_TextureMemory(int32_t begin_x, int32_t begin_y, int32_t end_x,
                                     int32_t end_y, cudaTextureObject_t tex, uint8_t* ret,
                                     int32_t width, int32_t cut_size_x, int32_t cut_size_y) {
  int32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  uint8_t* rety = ret + y * width;

  if (x < begin_x || x >= end_x || y < begin_y || y >= end_y) {
    *reinterpret_cast<uint32_t*>(rety + x) = 0;
    return;
  }

  int32_t shift = (end_x - x) * 8;
  uint8x4_t mask;
  mask.u32 = 0xFFFFFFFF ^ (0xFFFFFFFF << shift);

  for (int32_t scan_y = 0; scan_y < cut_size_y; scan_y++) {
    const int32_t pixel_y = y + scan_y;
    for (int32_t scan_x = 0; scan_x < cut_size_x; scan_x++) {
      const int32_t pixel_x = x + scan_x;
      mask.u8[0] &= tex2D<uint8_t>(tex, pixel_x, pixel_y);
      mask.u8[1] &= tex2D<uint8_t>(tex, pixel_x + 1, pixel_y);
      mask.u8[2] &= tex2D<uint8_t>(tex, pixel_x + 2, pixel_y);
      mask.u8[3] &= tex2D<uint8_t>(tex, pixel_x + 3, pixel_y);
    }
  }
  *reinterpret_cast<uint32_t*>(rety + x) = mask.u32;
}

__global__ void Set1(uint8_t* dev_ptr, int32_t row_bytes, uint8_t value) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

  dev_ptr[row_bytes * y + x] = value;
}

} // namespace gpu
} // namespace toybox
