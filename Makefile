NVFLAGS=-g -arch=compute_20 -code=sm_20 -O3 -lrt
# list .c and .cu source files here
SRCFILES=main.cpp ./Video/Renderer/CudaRenderer.cu ./Video/Renderer/cudaKernels.cu ./Video/Renderer/SoftwareRenderer.cpp

all:	Raytracer

Rasterizer: $(SRCFILES) 
	nvcc $(NVFLAGS) -o Raytracer $^ 

clean: 
	rm -rf *.o Raytracer
