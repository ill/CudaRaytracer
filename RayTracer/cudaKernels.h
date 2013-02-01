#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

#include <stdint.h>
#include "RayTracerCuda.h"

__global__ void RTkernel(Camera_t camera, uint32_t* colorBufferD, int xRes, int yRes);

#endif
