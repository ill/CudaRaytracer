#ifndef RAY_TRACER_CUDA_H_
#define RAY_TRACER_CUDA_H_

#include <vector>
#include "RayTracerBase.h"

#define BLOCK_WIDTH 10
#define BLOCK_HEIGHT 10


typedef struct Camera_t {
   glm::mat4 m_transform;
   //Frustum<> m_frustum;
   glm::mat4 m_modelView;
   glm::mat4 m_projection;
   glm::mat4 m_canonical;
} Camera_t;

typedef struct Scene {
   Camera_t camera;
   uint32_t* colorBuffer;
   int xRes;
   int yRes;
   RayTracerBase::SphereData* spheres;
   RayTracerBase::SphereData* sharedSpheres;
   int numSpheres;
   RayTracerBase::SphereData* lights;
   int numLights;
} Scene;

class RayTracerCuda : public RayTracerBase {
public:
    RayTracerCuda(const glm::uvec2& resolution);

    ~RayTracerCuda();
    
    void rayTraceScene(const illGraphics::Camera& camera) const;

    void output(const char * fileName) const;

    //RayTracerCuda::SphereData* sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const RayTracerBase::SphereData* omitSphere = NULL);

    std::vector<SphereData> m_spheres;
    std::vector<SphereData> m_lights;

};

#endif
