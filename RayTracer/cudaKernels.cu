#include "illEngine/Graphics/serial/Camera/Camera.h"
#include "cudaKernels.h"
#define T_VALUE glm::mediump_float

__device__ void getPickSegmentDev(Camera_t camera, const glm::vec2& windowCoords, const glm::ivec2& viewportCorner, const glm::ivec2& viewportDimensions, glm::vec3& ptADestination, glm::vec3& ptBDestination) {
   ptADestination = glm::unProject(glm::vec3(windowCoords, 0.0f), camera.m_modelView, camera.m_projection, glm::ivec4(viewportCorner, viewportDimensions));
   ptBDestination = glm::unProject(glm::vec3(windowCoords, 1.0f), camera.m_modelView, camera.m_projection, glm::ivec4(viewportCorner, viewportDimensions));
}

/**
* Checks for a ray intersection with the sphere.
*
* @param rayOrigin The origin of the ray
* @param rayDirection The direction of the ray
* @param distance If a collision happened, this is the closest distance down the ray where the intersection occured.
*
* @return Whether or not an intersection happened.  Check this before using the intersection distance.
*
* Code modified from here http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
*/
__device__ bool rayIntersection(glm::detail::tvec3<T_VALUE> rayOrigin, const glm::detail::tvec3<T_VALUE>& rayDirection,
      T_VALUE& distance, glm::detail::tvec3<T_VALUE> m_center, T_VALUE m_radius) {

        //put the ray into object space of the sphere (TODO: make sure this is correct)
        rayOrigin -= m_center;

        //Compute A, B and C coefficients
        T_VALUE a = glm::dot(rayDirection, rayDirection);
        T_VALUE b = (T_VALUE)2 * glm::dot(rayDirection, rayOrigin);
        T_VALUE c = glm::dot(rayOrigin, rayOrigin) - (m_radius * m_radius);

        //Find discriminant
        T_VALUE disc = b * b - (T_VALUE)4 * a * c;

        // if discriminant is negative there are no real roots, so return
        // false as ray misses sphere
        if (disc < 0) {
            return false;
        }

        // compute q as described above
        T_VALUE distSqrt = glm::sqrt(disc);
        T_VALUE q;
        if (b < 0) {
            q = (-b - distSqrt) / (T_VALUE)2;
        }
        else {
            q = (-b + distSqrt) / (T_VALUE)2;
        }

        // compute t0 and t1
        T_VALUE t0 = q / a;
        T_VALUE t1 = c / q;

        // make sure t0 is smaller than t1
        if(t0 > t1) {
            // if t0 is bigger than t1 swap them around
            T_VALUE temp = t0;
            t0 = t1;
            t1 = temp;
        }

        // if t1 is less than zero, the object is in the ray's negative direction
        // and consequently the ray misses the sphere
        if (t1 < (T_VALUE) 0) {
            return false;
        }

        // if t0 is less than zero, the intersection point is at t1
        if (t0 < (T_VALUE) 0) {
            distance = t1;
            return true;
        }
        // else the intersection point is at t0
        else
        {
            distance = t0;
            return true;
        }
}

__device__ RayTracerBase::SphereData* sphereRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance,
      Scene scene, const RayTracerBase::SphereData* omitSphere = NULL) {
   //find the closest sphere
   RayTracerBase::SphereData* closestSphere = NULL;

   int i = 0;
   for (i = 0; i < scene.numSpheres; i++) {
      RayTracerBase::SphereData& sphere = scene.spheres[i];

      glm::mediump_float thisDistance;

      //do a ray sphere intersection
      if(rayIntersection(rayOrigin, rayDirection, thisDistance, sphere.m_sphere.m_center, sphere.m_sphere.m_radius)) {
         //if this is the closest sphere so far, choose this bill
         if(&sphere != omitSphere &&(closestSphere == NULL || thisDistance < distance)) {
             distance = thisDistance;
             closestSphere = &sphere;
         }
      }
   }

   return closestSphere;
}

// Kernel:
__global__ void RTkernel(Scene scene) {
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   glm::vec3 a;
   glm::vec3 b;

   // Get the ray direction
   getPickSegmentDev(scene.camera, glm::vec2(x, y), glm::ivec2(0, 0), glm::ivec2(scene.xRes, scene.yRes), a, b);
   b = glm::normalize(b - a);
   
   glm::mediump_float distance;
   RayTracerBase::SphereData* sphere = sphereRay(a, b, distance, scene);

   // found a sphere
   if (sphere) {
      glm::vec4 m_finalColor = sphere->m_color * 0.1f;    //even if in shadow, give a bit of ambient lighting

      // get intersection position
      glm::vec3 intersection = a + b * distance;

      //for every light
      for (int i = 0; i < scene.numLights; i++) {
         RayTracerBase::SphereData& light = scene.lights[i];

         //get direction to light
         glm::vec3 lightDir = glm::normalize(light.m_sphere.m_center - intersection);

         //get distance squared to light
         //glm::mediump_float lightDistance2 = distance2(intersection, light.m_sphere.m_center);
      }
   }
}

