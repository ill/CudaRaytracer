#include "illEngine/Graphics/serial/Camera/Camera.h"

#include "outputTga.h"
#include "RayTracerCpu.h"
#include "util.h"

RayTracerCpu::RayTracerCpu(const glm::uvec2& resolution)
    : RayTracerBase(resolution)
{
    m_colorBuffer = new uint32_t[m_resolution.x * m_resolution.y];

    //create all the spheres
    for(unsigned int x = 0; x < 10; x++) {
        for(unsigned int y = 0; y < 10; y++) {
            for(unsigned int z = 0; z < 10; z++) {
                m_spheres.push_back(SphereData());

                m_spheres.back().m_color = glm::vec4(0.1f + (float) x / 10.0f, 0.1f + (float) y / 10.0f, 0.1f + (float) z / 10.0f, 1.0f);
                m_spheres.back().m_sphere.m_radius = 3.0f;
                m_spheres.back().m_sphere.m_center = glm::vec3(10.0f * x, 10.0f * y, 10.0f *z);
            }
        }
    }

    //create the lights

    //a light white light
    m_lights.push_back(SphereData());

    m_lights.back().m_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    m_lights.back().m_sphere.m_radius = 100.0f;
    m_lights.back().m_sphere.m_center = glm::vec3(-30.0f, 50.0f, 50.0f);

    //a red light
    m_lights.push_back(SphereData());

    m_lights.back().m_color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    m_lights.back().m_sphere.m_radius = 30.0f;
    m_lights.back().m_sphere.m_center = glm::vec3(50.0f, 50.0f, -10.0f);

    //a blue light
    m_lights.push_back(SphereData());

    m_lights.back().m_color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    m_lights.back().m_sphere.m_radius = 30.0f;
    m_lights.back().m_sphere.m_center = glm::vec3(50.0f, -10.0f, 50.0f);
}

RayTracerCpu::~RayTracerCpu() {
    delete[] m_colorBuffer;
}

const RayTracerCpu::SphereData* RayTracerCpu::sphereForRay(const glm::vec3& rayOrigin, const glm::vec3& rayDirection, glm::mediump_float& distance, const SphereData* omitSphere) const {
    //find the closest sphere
    const SphereData * closestSphere = NULL;

    for (std::vector<SphereData>::const_iterator iter = m_spheres.begin(); iter != m_spheres.end(); iter++) {
        const SphereData& sphere = *iter;

        glm::mediump_float thisDistance;

        //do a ray sphere intersection
        if(sphere.m_sphere.rayIntersection(rayOrigin, rayDirection, thisDistance)) {
            //if this is the closest sphere so far, choose this bill
            if(&sphere != omitSphere &&(closestSphere == NULL || thisDistance < distance)) {
                distance = thisDistance;
                closestSphere = &sphere;
            }
        }
    }

    return closestSphere;
}

void RayTracerCpu::rayTraceScene(const illGraphics::Camera& camera) const {
    for(unsigned int x = 0; x < m_resolution.x; x++) {
        for(unsigned int y = 0; y < m_resolution.y; y++) {
            glm::vec3 a;
            glm::vec3 b;

            camera.getPickSegment(glm::vec2(x, y), glm::ivec2(0, 0), glm::ivec2(m_resolution.x, m_resolution.y), a, b);

            //get the ray direction
            b = glm::normalize(b - a);

            glm::mediump_float distance;
            const SphereData * sphere = sphereForRay(a, b, distance);

            //found a sphere
            if(sphere) {
                glm::vec4 m_finalColor = sphere->m_color * 0.1f;    //even if in shadow, give a bit of ambient lighting

                //get intersection position
                glm::vec3 intersection = a + b * distance;

                //for every light
                for(std::vector<SphereData>::const_iterator iter = m_lights.begin(); iter != m_lights.end(); iter++) {
                    const SphereData& light = *iter;

                    //get direction to light
                    glm::vec3 lightDir = glm::normalize(light.m_sphere.m_center - intersection);

                    //get distance squared to light
                    glm::mediump_float lightDistance2 = distance2(intersection, light.m_sphere.m_center);
                    
                    //check if within light radius
                    if(lightDistance2 > light.m_sphere.m_radius * light.m_sphere.m_radius) {
                        continue;
                    }

                    //check if the light is in shadow
                    {
                        glm::mediump_float shadowDistance2;
                        const SphereData * shadowSphere = sphereForRay(intersection, lightDir, shadowDistance2, sphere);
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

                m_colorBuffer[x + m_resolution.x * y] = vecToInt(glm::clamp(m_finalColor, 0.0f, 1.0f));
            }
            else {
                m_colorBuffer[x + m_resolution.x * y] = 0;
            }
        }
    }
}

void RayTracerCpu::output(const char * fileName) const {
    tgaOut(m_colorBuffer, m_resolution, fileName);
}
