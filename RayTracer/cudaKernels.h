#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

#include <stdint.h>

const unsigned int BLK_WIDTH = 32;
const unsigned int BLK_HEIGHT = 16;

struct Face;

template <typename T, int dim>
struct Arr {
   T m_data[dim];
};

__global__ void clearKernel(
   /*glm::vec3*/uint32_t color, 
   /*glm::uvec2*/Arr<unsigned int, 2> resolution, 
   uint32_t* colorBuffer, 
   uint32_t* zBuffer);
   
__global__ void renderKernel(
   /*glm::vec3*/Arr<float, 3> lightVector, 
   /*glm::mat4*/Arr<float, 16> fullTransform, 
   /*glm::mat3*/Arr<float, 9> normal, 
   /*glm::uvec2*/Arr<unsigned int, 2> resolution, 
   const Face * mesh,
   unsigned int numTri,
   uint32_t* colorBuffer, 
   unsigned int* zBuffer);

__global__ void blurKernel(
    uint32_t *colorBuffer,
    uint32_t *colorBufferCopy,
    uint32_t *colorBufferThird,
    Arr<unsigned int, 2> resolution);

#endif
