#ifndef __COMMON_RENDER_H__
#define __COMMON_RENDER_H__

#ifdef USE_CUDA
#define DEVICE __device__
#else
#define DEVICE
#endif

//TODO: put the rayForSphere function in here

//TODO: all the raytracing code can go here


/**
 * The common rendering code between the CUDA and the software version goes here
 */
//per pixel lighting
//DEVICE inline glm::vec4 vertexShader(const glm::vec3& position, const glm::mat4& modelViewProjectionMatrix, glm::vec3& normal, const glm::mat3& normalMatrix) {
//  normal = normalMatrix * normal;
//  return modelViewProjectionMatrix * glm::vec4(position, 1.0f);
//}
//
//DEVICE inline uint32_t fragmentShader(const glm::vec3& color, const glm::vec3& normal, const glm::vec3& lightVector) {
//  //TODO: specular?  Otherwise this is identical to fragmentShaderOK
//
//  glm::mediump_float lightNormDot = glm::dot(lightVector, glm::normalize(normal));
//  glm::vec3 diffuse = color * glm::max(0.0f, lightNormDot);
//
//  return vecToInt(diffuse);
//}
//
////const static float offset[3] = {0.0, 1.3846153846, 3.2307692308};
//DEVICE const static int offset[5] = {0, 1, 2, 3, 4};
//DEVICE const static float weight[5] = {0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162};
//
//DEVICE inline uint32_t vertBlurShader(uint32_t* colorBuffer, const glm::uvec2& coord, const glm::uvec2& resolution) {
//  glm::vec3 color = intToVec(colorBuffer[coord.x + coord.y * resolution.x]) * weight[0];
//
//  for (int i=1; i<5; i++) {
//    int y = coord.y + offset[i];
//    if(y >= resolution.y) {
//      y = resolution.y - 1;
//    }
//
//    color += intToVec(colorBuffer[coord.x + y * resolution.x]) * weight[i];
//
//    y = coord.y - offset[i];
//    if(y < 0) {
//      y = 0;
//    }
//
//    color += intToVec(colorBuffer[coord.x + y * resolution.x]) * weight[i];
//  }
//
//  return vecToInt(color);
//}
//
//DEVICE inline uint32_t horzBlurShader(uint32_t* colorBuffer, const glm::uvec2& coord, const glm::uvec2& resolution) {
//  glm::vec3 color = intToVec(colorBuffer[coord.x + coord.y * resolution.x]) * weight[0];
//
//  for (int i=1; i<5; i++) {
//    int x = coord.x + offset[i];
//    if(x >= resolution.x) {
//      x = resolution.x - 1;
//    }
//
//    color += intToVec(colorBuffer[x + coord.y * resolution.x]) * weight[i];
//
//    x = coord.x - offset[i];
//    if(x < 0) {
//      x = 0;
//    }
//    
//    color += intToVec(colorBuffer[x + coord.y * resolution.x]) * weight[i];
//  }
//
//  return vecToInt(color);
//}
//
////GPU normally does this afterwards without the vertex shader
////set the coordinates to the normalized device coordinates and then to window space
//DEVICE inline glm::vec3 vtxClipToWindow(const glm::vec4& clipSpaceVertex, const glm::uvec2& resolution) {
//  return glm::vec3(
//      ((clipSpaceVertex.x / clipSpaceVertex.w) * 0.5f + 0.5f) * resolution.x,       //x
//      ((clipSpaceVertex.y / clipSpaceVertex.w) * 0.5f + 0.5f) * resolution.y,       //y
//      (1.0f + (clipSpaceVertex.z / clipSpaceVertex.w)) * 0.5f);                     //z
//}
//
//DEVICE inline Box<> triBoundingBox(const Face& face) {
//  Box<> res(face.m_vertex[0].m_pos);
//  res.addPoint(face.m_vertex[1].m_pos);
//  res.addPoint(face.m_vertex[2].m_pos);
//
//  return res;
//}
//
//DEVICE inline bool cullTriangle(const Box<>& boundingBox, glm::uvec2& resolution) {
//  return (boundingBox.m_max.x < -1.0f || boundingBox.m_min.x > resolution.x
//      || boundingBox.m_max.y < -1.0f || boundingBox.m_min.y > resolution.y
//      || boundingBox.m_max.z < -1.0f || boundingBox.m_min.z > 1.0f);
//}
//
//DEVICE inline void clipTriBoundingBox(Box<>& boundingBox, const glm::uvec2& resolution) {
//  Box<>(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(resolution.x - 1, resolution.y - 1, 1.0f)).constrain(boundingBox);
//}
//
////TODO: make the CUDA version not have branching
//DEVICE inline bool barycentricCoords(glm::vec3& dest, const Face& face, const glm::vec2& point) {
//  glm::mediump_float denom;
//
//  if((denom = implicitLine(glm::vec2(face.m_vertex[1].m_pos), glm::vec2(face.m_vertex[2].m_pos), glm::vec2(face.m_vertex[0].m_pos))) == 0.0f) {
//    return false;
//  }
//  dest.x = implicitLine(glm::vec2(face.m_vertex[1].m_pos), glm::vec2(face.m_vertex[2].m_pos), point) / denom;
//
//  if((denom = implicitLine(glm::vec2(face.m_vertex[0].m_pos), glm::vec2(face.m_vertex[2].m_pos), glm::vec2(face.m_vertex[1].m_pos))) == 0.0f) {
//    return false;
//  }
//  dest.y = implicitLine(glm::vec2(face.m_vertex[0].m_pos), glm::vec2(face.m_vertex[2].m_pos), point) / denom;
//
//  dest.z = 1.0f - dest.x - dest.y;
//
//  return true;
//}
//
//DEVICE inline bool barycentricCoordsInTri(const glm::vec3& baryCoords) {
//  return (baryCoords.x >= 0.0f && baryCoords.x <= 1.0f)
//    && (baryCoords.y >= 0.0f && baryCoords.y <= 1.0f)
//    && (baryCoords.z >= 0.0f && baryCoords.z <= 1.0f);
//}
//
//DEVICE inline uint32_t lerpDepth(const Face& face, const glm::vec3& baryCoords) {
//  return (baryCoords.x * face.m_vertex[0].m_pos.z +
//      baryCoords.y * face.m_vertex[1].m_pos.z +
//      baryCoords.z * face.m_vertex[2].m_pos.z) * UINT_MAX;  //TODO: make sure this is correct
//}
//
//DEVICE inline glm::vec3 lerpNorm(const Face& face, const glm::vec3& baryCoords) {
//  return baryCoords.x * face.m_vertex[0].m_norm +
//    baryCoords.y * face.m_vertex[1].m_norm +
//    baryCoords.z * face.m_vertex[2].m_norm;
//}

#endif
