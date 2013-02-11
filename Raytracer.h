#ifndef RAYTRACER_H_
#define RAYTRACER_H_

#include <glm/glm.hpp>

struct SphereData {
   glm::vec4 m_color;
   glm::mediump_float m_radius;
   glm::vec3 m_center;
};

typedef struct Camera_t {
   glm::mat4 m_transform;
   //Frustum<> m_frustum;
   glm::mat4 m_modelView;
   glm::mat4 m_projection;
   glm::mat4 m_canonical;
} Camera_t;

typedef struct Scene {
   SphereData* spheres;
   SphereData* sharedSpheres;
   int numSpheres;
   SphereData* lights;
   int numLights;
} Scene;

#endif
