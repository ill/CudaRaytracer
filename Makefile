CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -Iglm-0.9.4.1 -IillEngine -I.
LDFLAGS= -O3  -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -Iglm-0.9.4.1 -IillEngine -I.
CUDAFLAGS= -O3 -c -arch=sm_21 -Iglm-0.9.4.1 -IillEngine -I.

ALL= CameraController.o callbacksPBO.o kernelPBO.o simpleGLmain.o simplePBO.o

all= $(ALL) RTRT

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o RTRT

CameraController.o:	CameraController.cpp
	$(CC) $(CFLAGS) -o $@ $<

callbacksPBO.o:	callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

kernelPBO.o:	kernelPBO.cu
	$(CC) $(CUDAFLAGS) -o $@ $<

simpleGLmain.o:	simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

