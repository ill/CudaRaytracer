#ifndef RAY_TRACER_CUDA_H_
#define RAY_TRACER_CUDA_H_

#include <vector>
#include "RayTracerBase.h"

class RayTracerCuda : public RayTracerBase {
public:
    RayTracerCuda(const glm::uvec2& resolution);

    ~RayTracerCuda();
    
    void rayTraceScene(const illGraphics::Camera& camera) const;

    void output(const char * fileName) const;
};

#endif