CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lGLEW -lglut -Iglew-1.9.0/include
LDFLAGS= -O3  -lGL -lGLEW  -lglut -Iglew-1.9.0/include
CUDAFLAGS= -O3 -c -arch=sm_21 -Iglew-1.9.0/include

ALL= callbacksPBO.o kernelPBO.o simpleGLmain.o simplePBO.o

all= $(ALL) RTRT

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o RTRT

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

