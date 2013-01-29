#ifndef UTIL_H_
#define UTIL_H_

#include <glm/glm.hpp>
#include <stdint.h>

#ifdef USE_CUDA
#define DEVICE __device__
#else
#define DEVICE
#endif

DEVICE inline uint32_t vecToInt(const glm::vec4& vec) {
   uint32_t res = 0;

   res |= (uint8_t)(vec.x * 255) << 24;
   res |= (uint8_t)(vec.y * 255) << 16;
   res |= (uint8_t)(vec.z * 255) << 8;
   res |= (uint8_t)(vec.w * 255);
   
   return res;
}

DEVICE inline glm::vec4 intToVec(uint32_t color) {
   glm::vec4 res;

   res.x = ((color & 0xFF000000) >> 24) * (1.0f/255.0f);
   res.y = ((color & 0x00FF0000) >> 16) * (1.0f/255.0f);
   res.z = ((color & 0x0000FF00) >> 8) * (1.0f/255.0f);
   res.w = ((color & 0x000000FF)) * (1.0f/255.0f);

   return res;
}

#endif