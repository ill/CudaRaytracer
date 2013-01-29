#ifndef RAY_TRACER_CPU_H_
#define RAY_TRACER_CPU_H_

#include <vector>
#include "RayTracerBase.h"

class RayTracerCpu : public RayTracerBase {
public:
    RayTracerCpu(const glm::uvec2& resolution);

    ~RayTracerCpu();
    
    void rayTraceScene(const illGraphics::Camera& camera) const;

    void output(const char * fileName) const;

    //TODO: move this into a common function
    const SphereData* sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const SphereData* omitSphere = NULL) const;
        
    //TODO: make these arrays
    std::vector<SphereData> m_spheres;
    std::vector<SphereData> m_lights;
};

#endif