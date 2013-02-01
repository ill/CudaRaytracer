
#include "illEngine/Graphics/serial/Camera/Camera.h"
#include "cudaKernels.h"

__device__ void getPickSegmentDev(Camera_t camera, const glm::vec2& windowCoords, const glm::ivec2& viewportCorner, const glm::ivec2& viewportDimensions, glm::vec3& ptADestination, glm::vec3& ptBDestination) {
   ptADestination = glm::unProject(glm::vec3(windowCoords, 0.0f), camera.m_modelView, camera.m_projection, glm::ivec4(viewportCorner, viewportDimensions));
   ptBDestination = glm::unProject(glm::vec3(windowCoords, 1.0f), camera.m_modelView, camera.m_projection, glm::ivec4(viewportCorner, viewportDimensions));
}


__global__ void RTkernel(Camera_t camera, uint32_t* colorBufferD, int xRes, int yRes) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   glm::vec3 a;
   glm::vec3 b;

   glm::ivec2 c;
   c = glm::ivec2(xRes, yRes);

   // Get the ray direction
   getPickSegmentDev(camera, glm::vec2(x, y), glm::ivec2(0, 0), glm::ivec2(xRes, yRes), a, b);
   
//   a = glm::unProject(glm::vec3(glm::vec2(x, y), 0.0f), camera.getModelView(), camera.getProjection(), glm::ivec4(glm::ivec2(0, 0), glm::ivec2(m_resolution.x, m_resolution.y)));
//   b = glm::unProject(glm::vec3(glm::vec2(x, y), 1.0f), camera.getModelView(), camera.getProjection(), glm::ivec4(glm::ivec2(0, 0), glm::ivec2(m_resolution.x, m_resolution.y)));
//   b = normalize(b - a);
}

