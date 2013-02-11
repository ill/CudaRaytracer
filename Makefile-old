NVFLAGS=-g -arch=compute_20 -code=sm_20 -O3 -lrt
# list .c and .cu source files here
SRCFILES=main.cpp RayTracer/RayTracerCuda.cu RayTracer/cudaKernels.cu RayTracer/RayTracerCpu.cpp

all:	Raytracer

Raytracer: $(SRCFILES) 
	nvcc $(NVFLAGS) -o Raytracer -Iglm-0.9.4.1 -IillEngine -I. $^

clean: 
	rm -f *.o Raytracer
