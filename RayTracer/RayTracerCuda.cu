#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include <stdio.h>

#include "illEngine/Graphics/serial/Camera/Camera.h"

#include "outputTga.h"
#include "RayTracerCuda.h"
#include "util.h"
#include "cudaKernels.h"

RayTracerCuda::RayTracerCuda(const glm::uvec2& resolution) : RayTracerBase(resolution) {

    m_colorBuffer = new uint32_t[m_resolution.x * m_resolution.y];

    //create all the spheres
	for(unsigned int sphere = 0; sphere < 100; sphere++) {
		m_spheres.push_back(SphereData());

		m_spheres.back().m_color = glm::linearRand(glm::vec4(0.2f), glm::vec4(1.0f));
		m_spheres.back().m_sphere.m_radius = glm::linearRand(2.0f, 10.0f);
		m_spheres.back().m_sphere.m_center = glm::linearRand(glm::vec3(0.0f), glm::vec3(200.0f));
	}
	
    //create the lights
	m_lights.push_back(SphereData());

	m_lights.back().m_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	m_lights.back().m_sphere.m_radius = 300.0f;
	m_lights.back().m_sphere.m_center = glm::vec3(-30.0f, 50.0f, 50.0f);
}

RayTracerCuda::~RayTracerCuda() {
   delete[] m_colorBuffer;
}


/*__device__ RayTracerCuda::SphereData * RayTracerCuda::sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const SphereData* omitSphere) {
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
}*/

void RayTracerCuda::rayTraceScene(const illGraphics::Camera& camera) const {
   uint32_t* colorBufferD;
   RayTracerBase::SphereData* spheresD;
   RayTracerBase::SphereData* lightsD;
   const RayTracerBase::SphereData* tmpSpheres = &m_spheres[0];
   const RayTracerBase::SphereData* tmpLights = &m_lights[0];
   glm::vec3 a;
   glm::vec3 b;
   Scene scene;

   // Allocate device memory
   cudaMalloc((void **)&colorBufferD, m_resolution.x * m_resolution.y * sizeof(uint32_t));
   cudaMalloc((void **)&spheresD, m_spheres.size() * sizeof(SphereData));
   cudaMalloc((void **)&lightsD, m_lights.size() * sizeof(SphereData));
   cudaMemcpy(spheresD, tmpSpheres, m_spheres.size() * sizeof(SphereData), cudaMemcpyHostToDevice);
   cudaMemcpy(lightsD, tmpLights, m_lights.size() * sizeof(SphereData), cudaMemcpyHostToDevice);

   // Initialize things for kernel
   Camera_t camera_t;
   camera_t.m_transform = camera.getTransform();
   //camera_t.m_frustum = camera.getViewFrustum();
   camera_t.m_modelView = camera.getModelView();
   camera_t.m_projection = camera.getProjection();
   camera_t.m_canonical = camera.getCanonical();
   scene.camera = camera_t;
   scene.colorBuffer = colorBufferD;
   scene.xRes = m_resolution.x;
   scene.yRes = m_resolution.y;
   scene.spheres = spheresD;
   scene.numSpheres = m_spheres.size();
   scene.lights = lightsD;
   scene.numLights = m_lights.size();

   // Set up grid and block dimensions
   dim3 dimGrid(ceil((float) m_resolution.x / BLOCK_WIDTH), ceil((float) m_resolution.y / BLOCK_HEIGHT));
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

   //dim3 dimGrid(10, 10);
   //dim3 dimBlock(10, 10);
   
   // Call kernel
   //RTkernel<<<dimGrid, dimBlock>>>(camera_t, colorBufferD, m_resolution.x, m_resolution.y, spheresD, lightsD);
   RTkernel<<<dimGrid, dimBlock>>>(scene);
   
   // make the host block until the device is finished with foo
   cudaThreadSynchronize();

   // check for error
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess) {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
   }

   // Retrieve results
   cudaMemcpy(m_colorBuffer, colorBufferD, m_resolution.x * m_resolution.y * sizeof(uint32_t), cudaMemcpyDeviceToHost);

   // Clean up, free data from global memory
   cudaFree(colorBufferD);
   cudaFree(spheresD);
   cudaFree(lightsD);
}

void RayTracerCuda::output(const char * fileName) const {
    tgaOut(m_colorBuffer, m_resolution, fileName);
}
