#include "illEngine/Graphics/serial/Camera/Camera.h"
#include "cudaKernels.h"
#include "util.h"

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

__device__ T_VALUE dist2(const glm::detail::tvec3<T_VALUE>& v1, const glm::detail::tvec3<T_VALUE>& v2) {
    return (v1.x - v2.x) * (v1.x - v2.x) 
        + (v1.y - v2.y) * (v1.y - v2.y)
        + (v1.z - v2.z) * (v1.z - v2.z);
}

__device__ uint32_t vecToIntD(const glm::vec4& vec) {
   uint32_t res = 0;

   res |= (uint8_t)(vec.x * 255) << 24;
   res |= (uint8_t)(vec.y * 255) << 16;
   res |= (uint8_t)(vec.z * 255) << 8;
   res |= (uint8_t)(vec.w * 255);
   
   return res;
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
         glm::mediump_float lightDistance2 = dist2(intersection, light.m_sphere.m_center);

         //check if within light radius
         if (lightDistance2 > light.m_sphere.m_radius * light.m_sphere.m_radius) {
             continue;
         }

         //check if the light is in shadow
         {
            glm::mediump_float shadowDistance2;
            RayTracerBase::SphereData* shadowSphere = sphereRay(intersection, lightDir, shadowDistance2, scene, sphere);
            shadowDistance2 *= shadowDistance2;
 
            if(shadowSphere && shadowDistance2 <= lightDistance2) {
                continue;
            }
        }

        //half vector
        glm::vec3 halfVec = glm::normalize(lightDir + glm::normalize(b));

        glm::mediump_float lightDistance = glm::sqrt(lightDistance2);

        //light attenuation
        glm::mediump_float attenuation = glm::clamp((light.m_sphere.m_radius - lightDistance) / light.m_sphere.m_radius, 0.0f, 1.0f);

        //get normal of sphere at intersection
        glm::vec3 normal = glm::normalize(intersection - sphere->m_sphere.m_center);

        glm::mediump_float lambertFactor = glm::max(0.0f, glm::dot(lightDir, normal));

        //now add the diffuse contribution
        m_finalColor += attenuation * lambertFactor * light.m_color * sphere->m_color;

        //now add the specular contribution (Meh, maybe later)
        //m_finalColor += glm::max(glm::vec4(0.0f), glm::pow(glm::max(0.0f, glm::dot(halfVec, normal)), 5.0f) * sphere->m_color * light.m_color * glm::clamp(lambertFactor * 4.0f, 0.0f, 1.0f));
      }

      scene.colorBuffer[x + scene.xRes * y] = vecToIntD(glm::clamp(m_finalColor, 0.0f, 1.0f));
   }
   else {
      scene.colorBuffer[x + scene.xRes * y] = 0;
   }
}

