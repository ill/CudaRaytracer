#ifndef RAY_TRACER_CUDA_H_
#define RAY_TRACER_CUDA_H_

#include <vector>
#include "RayTracerBase.h"
#include "Util/Geometry/Frustum.h"

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

typedef struct Camera_t {
   glm::mat4 m_transform;
   //Frustum<> m_frustum;
   glm::mat4 m_modelView;
   glm::mat4 m_projection;
   glm::mat4 m_canonical;
} Camera_t;

class RayTracerCuda : public RayTracerBase {
public:
    RayTracerCuda(const glm::uvec2& resolution);

    ~RayTracerCuda();
    
    void rayTraceScene(const illGraphics::Camera& camera) const;

    void output(const char * fileName) const;

    const SphereData* sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const SphereData* omitSphere = NULL) const;

    std::vector<SphereData> m_spheres;
    std::vector<SphereData> m_lights;

};

#endif
