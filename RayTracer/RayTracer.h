#ifndef RAY_TRACER_H_
#define RAY_TRACER_H_

#include <vector>
#include "Util/Geometry/Sphere.h"

namespace illGraphics {
    class Camera;
}

class RayTracer {
public:
    struct SphereData {
        glm::vec4 m_color;
        Sphere<> m_sphere;
    };

    RayTracer(const glm::uvec2& resolution);

    ~RayTracer();
    
    void rayTraceScene(const illGraphics::Camera& camera) const;

    void output(const char * fileName) const;

    const SphereData* sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const SphereData* omitSphere = NULL) const;
        
    glm::uvec2 m_resolution;
    uint32_t * m_colorBuffer;

    std::vector<SphereData> m_spheres;
    std::vector<SphereData> m_lights;
};

#endif