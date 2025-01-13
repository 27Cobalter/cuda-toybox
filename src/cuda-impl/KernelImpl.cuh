#include <cuda_runtime.h>

namespace toybox {
namespace gpu {
union uint8x4_t {
  uint32_t u32;
  uint8_t u8[4];
};

__global__ void Kernel_Naive(int32_t begin_x, int32_t begin_y, int32_t end_x, int32_t end_y,
                             uint8_t* src, uint8_t* ret, int32_t width, int32_t cut_size_x,
                             int32_t cut_size_y);

__global__ void Kernel_U32Simd(int32_t begin_x, int32_t begin_y, int32_t end_x, int32_t end_y,
                               uint8_t* src, uint8_t* ret, int32_t width, int32_t cut_size_x,
                               int32_t cut_size_y);

__global__ void Kernel_TextureMemory(int32_t begin_x, int32_t begin_y, int32_t end_x,
                                     int32_t end_y, cudaTextureObject_t tex, uint8_t* ret,
                                     int32_t width, int32_t cut_size_x, int32_t cut_size_y);

} // namespace gpu
} // namespace toybox
