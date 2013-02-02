#ifndef RAY_TRACER_BACE_H_
#define RAY_TRACER_BACE_H_

#include "Util/Geometry/Sphere.h"



namespace illGraphics {
    class Camera;
}

class RayTracerBase {
public:

    struct SphereData {
      glm::vec4 m_color;
      Sphere<> m_sphere;
    };

    RayTracerBase(glm::uvec2 resolution)
        : m_resolution(resolution)
    {}
        
    virtual void rayTraceScene(const illGraphics::Camera& camera) const = 0;
    virtual void output(const char * fileName) const = 0;
                
    glm::uvec2 m_resolution;
    uint32_t * m_colorBuffer;
};

#endif
