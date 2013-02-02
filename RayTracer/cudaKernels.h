#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

#include <stdint.h>
#include "RayTracerCuda.h"

__device__ void getPickSegmentDev(Camera_t camera, const glm::vec2& windowCoords, const glm::ivec2& viewportCorner, const glm::ivec2& viewportDimensions, glm::vec3& ptADestination, glm::vec3& ptBDestination);
__global__ void RTkernel(Scene scene);

#endif
