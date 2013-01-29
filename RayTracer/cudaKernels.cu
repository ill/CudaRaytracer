#include "cudaKernels.h"

#define USE_CUDA
#include "../../Util/Geometry/geomUtil.h"
#include "../../Util/Geometry/Box.h"
#include "../../Util/Geometry/Mesh.h"
#include "commonRender.h"

__global__ void clearKernel(
    /*glm::vec3*/uint32_t color,
    /*glm::uvec2*/Arr<unsigned int, 2> resolution,
    uint32_t* colorBuffer,
    unsigned int *zBuffer)
{
  int x = threadIdx.x + BLK_WIDTH * blockIdx.x;
  int y = threadIdx.y + BLK_HEIGHT * blockIdx.y;

  if (x < ((glm::uvec2&) resolution).x && y < ((glm::uvec2&) resolution).y)
  {
    colorBuffer[y * ((glm::uvec2&) resolution).x + x] = color;
    zBuffer[y * ((glm::uvec2&) resolution).x + x] = UINT_MAX;
  }
}

__global__ void renderKernel(
    /*glm::vec3*/Arr<float, 3> lightVector,
    /*glm::mat4*/Arr<float, 16> fullTransform,
    /*glm::mat3*/Arr<float, 9> normal,
    /*glm::uvec2*/Arr<unsigned int, 2> resolution,
    const Face * mesh,
    unsigned int numTri,
    uint32_t* colorBuffer,
    unsigned int *zBuffer)
{
  unsigned int triIndex = blockIdx.x + blockIdx.y * gridDim.x;

  if (triIndex > numTri - 1) {
    return;
  }

  Face face = mesh[triIndex];

  //step 1: vertex shader
  for(unsigned int vertexInd = 0; vertexInd < 3; vertexInd++) {
    //code based on this: http://www.opengl.org/wiki/Vertex_Transformation

    Vertex& vertex = face.m_vertex[vertexInd];

    vertex.m_pos = vtxClipToWindow(vertexShader(vertex.m_pos, (glm::mat4 &) fullTransform, vertex.m_norm, (glm::mat3 &) normal), (glm::uvec2 &) resolution);
  }

  //step 2: fragment shader
  //for(unsigned int triangleInd = 0; triangleInd < meshCopy.m_numTri; triangleInd++) {
  //figure out triangle bounding box
  Box<> boundingBox = triBoundingBox(face);

  //cull triangle
  if(cullTriangle(boundingBox, (glm::uvec2 &) resolution)) {
    //continue;
  }

  //clip bounding box
  clipTriBoundingBox(boundingBox, (glm::uvec2 &) resolution);

  //draw the triangle
  for(unsigned int x = (unsigned int) boundingBox.m_min.x; x <= (unsigned int) boundingBox.m_max.x; x++) {
    for(unsigned int y = (unsigned int) boundingBox.m_min.y; y <= (unsigned int) boundingBox.m_max.y; y++) {
      //m_colorBuffer[x + m_resolution.x * y] = glm::vec3(1.0f, 1.0f, 1.0f);
      //continue;

      //check if point is in triangle using barycentric coords
      //bool writePixel = false;   //to avoid divergence

      glm::vec3 bary;

      if(!barycentricCoords(bary, face, glm::vec2(x, y))) {
        continue;
      }

      if(barycentricCoordsInTri(bary)) {
        unsigned int depth = lerpDepth(face, bary);

        glm::vec3 norm = lerpNorm(face, bary);
        uint32_t color = fragmentShader(glm::vec3(0.0f, 0.0f, 1.0f), norm, (glm::vec3 &) lightVector);

        if(atomicMin(&zBuffer[x + ((glm::uvec2 &) resolution).x * y], depth) > depth) {
          colorBuffer[x + ((glm::uvec2 &) resolution).x * y] = color;      //Warning, this isn't 100% atomic
        }
      }
      else {
        continue;
      }
    }
  }
}

__global__ void blurKernel(uint32_t *colorBuffer, uint32_t *colorBufferCopy, uint32_t *colorBufferThird, Arr<unsigned int, 2> resolution)
{

  int x = threadIdx.x + BLK_WIDTH * blockIdx.x;
  int y = threadIdx.y + BLK_HEIGHT * blockIdx.y;

  if (x < ((glm::uvec2&) resolution).x && y < ((glm::uvec2&) resolution).y)
  {
      colorBufferCopy[x+((glm::uvec2&)resolution).x  * y] = vertBlurShader(colorBuffer, glm::uvec2(x,y), (glm::uvec2 &)resolution);
      colorBufferThird[x+((glm::uvec2&)resolution).x  * y] = horzBlurShader(colorBufferCopy, glm::uvec2(x,y), (glm::uvec2  &)resolution);
  }

}


