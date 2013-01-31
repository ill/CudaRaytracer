#include "illEngine/Graphics/serial/Camera/Camera.h"

#include "outputTga.h"
#include "RayTracerCuda.h"
#include "util.h"



RayTracerCuda::RayTracerCuda(const glm::uvec2& resolution) : RayTracerBase(resolution) {

    m_colorBuffer = new uint32_t[m_resolution.x * m_resolution.y];

    //create all the spheres
    for(unsigned int x = 0; x < 10; x++) {
        for(unsigned int y = 0; y < 10; y++) {
            for(unsigned int z = 0; z < 10; z++) {
                m_spheres.push_back(SphereData());

                m_spheres.back().m_color = glm::vec4(0.1f + (float) x / 10.0f, 0.1f + (float) y / 10.0f, 0.1f + (float) z / 10.0f, 1.0f);
                m_spheres.back().m_sphere.m_radius = 3.0f;
                m_spheres.back().m_sphere.m_center = glm::vec3(10.0f * x, 10.0f * y, 10.0f *z);
            }
        }
    }

    //create the lights

    //a light white light
    m_lights.push_back(SphereData());

    m_lights.back().m_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    m_lights.back().m_sphere.m_radius = 100.0f;
    m_lights.back().m_sphere.m_center = glm::vec3(-30.0f, 50.0f, 50.0f);

    //a red light
    m_lights.push_back(SphereData());

    m_lights.back().m_color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    m_lights.back().m_sphere.m_radius = 30.0f;
    m_lights.back().m_sphere.m_center = glm::vec3(50.0f, 50.0f, -10.0f);

    //a blue light
    m_lights.push_back(SphereData());

    m_lights.back().m_color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    m_lights.back().m_sphere.m_radius = 30.0f;
    m_lights.back().m_sphere.m_center = glm::vec3(50.0f, -10.0f, 50.0f);
}

RayTracerCuda::~RayTracerCuda() {
   delete[] m_colorBuffer;
}

const RayTracerCuda::SphereData* RayTracerCuda::sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const SphereData* omitSphere) const {
    //find the closest sphere
    const SphereData * closestSphere = NULL;

    for (std::vector<SphereData>::const_iterator iter = m_spheres.begin(); iter != m_spheres.end(); iter++) {
        const SphereData& sphere = *iter;

        glm::mediump_float thisDistance;

        //do a ray sphere intersection
        if(sphere.m_sphere.rayIntersection(rayOrigin, rayDirection, thisDistance)) {
            //if this is the closest sphere so far, choose this bill
            if(&sphere != omitSphere &&(closestSphere == NULL || thisDistance < distance)) {
                distance = thisDistance;
                closestSphere = &sphere;
            }
        }
    }

    return closestSphere;
}

void RayTracerCuda::rayTraceScene(const illGraphics::Camera& camera) const {
   uint32_t* colorBufferD;

   // Allocate device memory
   cudaMalloc((void **)&colorBufferD, m_resolution.x * m_resolution.y * sizeof(uint32_t));

   // Set up grid and block dimensions
   dim3 dimGrid(ceil(m_resolution.x / 32), ceil(m_resolution.y / 32)); // TODO: ceil of res.x/32 and ceil of res.y/32
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

   // Call kernel
   //RTkernel<<<dimGrid, dimBlock>>>();

   // Retrieve results
   cudaMemcpy(m_colorBuffer, colorBufferD, m_resolution.x * m_resolution.y * sizeof(uint32_t), cudaMemcpyDeviceToHost);

   // Clean up, free data from global memory
}

void RayTracerCuda::output(const char * fileName) const {
    tgaOut(m_colorBuffer, m_resolution, fileName);
}
