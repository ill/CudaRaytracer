CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lGLEW -lglut 
LDFLAGS= -O3  -lGL -lGLEW  -lglut 
CUDAFLAGS= -O3 -c -arch=sm_21

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

