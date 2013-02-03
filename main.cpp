#include <glm/glm.hpp>
#include <cstring>
#include <cstdio>
#include <iostream>

#include "illEngine/Graphics/serial/Camera/Camera.h"
#include "illEngine/Util/Geometry/geomUtil.h"
#include "RayTracer/RayTracerCpu.h"

#define ENABLE_CUDA

#ifdef ENABLE_CUDA
#include "RayTracer/RayTracerCuda.h"
#endif

using namespace std;

int main(int argc, const char* argv[]) {
    bool useCuda = true;
    glm::uvec2 resolution (1000u, 1000u);
    std::string outputFile = "Output.tga";
    glm::mediump_float fieldOfView = 60;//illGraphics::DEFAULT_FOV;

    RayTracerBase * rayTracer;

    //parse some args
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "+W", 2) == 0)
            resolution.x = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "+H", 2) == 0)
            resolution.y = atoi(argv[i] + 2);
        else if (strncmp(argv[i], "+Cpu", 4) == 0)
            useCuda = false;
        else if (strncmp(argv[i], "+O", 2) == 0)
            outputFile = (argv[i] + 2);
        else if (strncmp(argv[i], "+Fov", 4) == 0)
            fieldOfView = (glm::mediump_float) atoi(argv[i] + 4);
        else
            printf("Unknown option: %s\n", argv[i]);
    }

#ifdef ENABLE_CUDA
    if(useCuda) {
        rayTracer = new RayTracerCuda(resolution);
    }
    else {
        rayTracer = new RayTracerCpu(resolution);
    }
#else
    rayTracer = new RayTracerCpu(resolution);
#endif 
    
    illGraphics::Camera camera;
    camera.setPerspectiveTransform(
		createTransform(glm::vec3(-50.0f, -50.0f, -50.0f), directionToMat3(glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f)))),
        //createTransform(glm::vec3(-50.0f, 0.0f, 0.0f), directionToMat3(glm::normalize(glm::vec3(2.0f, 1.0f, 1.0f)))),
        //createTransform(glm::vec3(-30.0f, 50.0f, 50.0f), directionToMat3(glm::normalize(glm::vec3(1.0f, 0.0f, 0.0f)))),
        (float) resolution.x / (float) resolution.y, fieldOfView);

    rayTracer->rayTraceScene(camera);
    rayTracer->output(outputFile.c_str());

    delete rayTracer;

	fflush(stdout);
	
    return 0;
}
