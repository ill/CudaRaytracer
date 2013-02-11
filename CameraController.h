#ifndef __CAMERA_CONTROLLER_H__
#define __CAMERA_CONTROLLER_H__

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "illEngine/Util/Geometry/geomUtil.h"

struct CameraController {
    CameraController();
    ~CameraController() {}

    void update(double seconds);

    glm::vec3 m_eulerAngles;
    glm::mat4 m_transform;

    float m_speed;
    float m_rollSpeed;

    bool m_forward;
    bool m_back;
    bool m_left;
    bool m_right;
    bool m_up;
    bool m_down;
    bool m_rollLeft;
    bool m_rollRight;
    bool m_sprint;

    bool m_lookMode;

    glm::mediump_float m_zoom;
};

#endif

