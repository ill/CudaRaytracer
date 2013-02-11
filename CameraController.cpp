#include "CameraController.h"

CameraController::CameraController()
    : m_speed(0.0f),
    m_rollSpeed(0.0f),

    m_forward(false),
    m_back(false),
    m_left(false),
    m_right(false),
    m_up(false),
    m_down(false),
    m_rollLeft(false),
    m_rollRight(false),
    m_sprint(false),

    m_lookMode(true),

    m_zoom(1.0f)
{}

void CameraController::update(double seconds) {
    glm::vec3 velocity(0.0f);

    if(m_forward) {
        velocity.z = -1.0f;
    }
    else if(m_back) {
        velocity.z = 1.0f;
    }

    if(m_left) {
        velocity.x = -1.0f;
    }
    else if(m_right) {
        velocity.x = 1.0f;
    }

    if(m_up) {
        velocity.y = 1.0f;
    }
    else if(m_down) {
        velocity.y = -1.0f;
    }      

    float speedMultiplier = m_sprint ? 5.0f : 1.0f;

    velocity = safeNormalize(velocity);

    velocity *= seconds * m_speed * speedMultiplier;

    if(m_lookMode) {        //quaternion mode
        if(m_rollLeft) {
            m_transform = glm::rotate(m_transform, (float) seconds * m_rollSpeed * speedMultiplier, glm::vec3(0.0f, 0.0f, 1.0f));
        }
        else if(m_rollRight) {
            m_transform = glm::rotate(m_transform, (float) seconds * -m_rollSpeed * speedMultiplier, glm::vec3(0.0f, 0.0f, 1.0f));
        }

        m_transform = glm::translate(m_transform, velocity);
    }
    else {                  //eueler mode
        if(m_rollLeft) {
            m_eulerAngles.z += (float) seconds * m_rollSpeed * speedMultiplier;
        }
        else if(m_rollRight) {
            m_eulerAngles.z -= (float) seconds * m_rollSpeed * speedMultiplier;
        }
            
        //Who the hell decided to make this function take them in this order!
        glm::vec3 position = getTransformPosition(m_transform);
        glm::mat4 rotation = glm::yawPitchRoll(glm::radians(m_eulerAngles.y), glm::radians(m_eulerAngles.x), glm::radians(m_eulerAngles.z));
            
        //TODO: make this not suck

        //vertical
        position.y += velocity.y;

        //forward
            
        glm::mediump_float rad = glm::radians(m_eulerAngles.y - 90);
            
        position.x += velocity.z * glm::cos(rad);
        position.z -= velocity.z * glm::sin(rad);
            
        //strafe
        rad = glm::radians(m_eulerAngles.y);
            
        position.x += velocity.x * glm::cos(rad);
        position.z -= velocity.x * glm::sin(rad);

        m_transform = glm::translate(position);
        m_transform = m_transform * rotation;
    }
}
