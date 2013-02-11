//callbacksPBO.cpp (Rob Farber)

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>

#include "illEngine/Graphics/serial/Camera/Camera.h"
#include "illEngine/Util/Geometry/geomUtil.h"

#include "CameraController.h"

CameraController cameraController;
extern illGraphics::Camera illCamera;
illGraphics::Camera illCamera;

int lastMouseX = 0;
int lastMouseY = 0;

// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;
extern void moveIn();
extern void moveOut();
extern void moveUp();
extern void moveDown();
extern void moveLeft();
extern void moveRight();

// The user must create the following routines:
void runCuda();

void display()
{   
   cameraController.m_speed = 1.0f;
   cameraController.m_rollSpeed = 1.0f;
   cameraController.update(1.0f);   
   
   illCamera.setPerspectiveTransform(cameraController.m_transform, (float) image_width / (float) image_height, cameraController.m_zoom * 90.0f);

   // run CUDA kernel
   runCuda();

   // Create a texture from the buffer
   glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

   // bind texture from PBO
   glBindTexture(GL_TEXTURE_2D, textureID);


   // Note: glTexSubImage2D will perform a format conversion if the
   // buffer is a different format from the texture. We created the
   // texture with format GL_RGBA8. In glTexSubImage2D we specified
   // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

   // Note: NULL indicates the data resides in device memory
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
         GL_RGBA, GL_UNSIGNED_BYTE, NULL);


   // Draw a single Quad with texture coordinates for each vertex.

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   // Don't forget to swap the buffers!
   glutSwapBuffers();

   // if animFlag is true, then indicate the display needs to be redrawn
   if(animFlag) {
      glutPostRedisplay();
      animTime += animInc;
   }
}

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
   printf("Key Down %c\n", key);

   switch(key) {
   case 'a':
      cameraController.m_left = true;
      break;
      
   case 'd':
      cameraController.m_right = true;
      break;
      
   case 'w':
      cameraController.m_forward = true;
      break;
      
   case 's':
      cameraController.m_back = true;
      break;
      
   case ' ':
      cameraController.m_up = true;
      break;
      
   case 'c':
      cameraController.m_down = true;
      break;
      
   case 'q':
      cameraController.m_rollLeft = true;
      break;
      
   case 'e':
      cameraController.m_rollRight = true;
      break; 
   
   case(27) :
      exit(0);
      break;
   }
}

void keyboardUp(unsigned char key, int x, int y)
{
   printf("Key Up %c\n", key);

   switch(key) {
   case 'a':
      cameraController.m_left = false;
      break;
      
   case 'd':
      cameraController.m_right = false;
      break;
      
   case 'w':
      cameraController.m_forward = false;
      break;
      
   case 's':
      cameraController.m_back = false;
      break;
      
   case ' ':
      cameraController.m_up = false;
      break;
      
   case 'c':
      cameraController.m_down = false;
      break;
      
   case 'q':
      cameraController.m_rollLeft = false;
      break;
      
   case 'e':
      cameraController.m_rollRight = false;
      break; 
   }
}

void mouseLook(int x, int y) {
   float xLook = (float) (x - lastMouseX);
   float yLook = -(float) (y - lastMouseY);
   
   lastMouseX = x;
   lastMouseY = y;

   printf("Mouse (%f, %f)\n", xLook, yLook);

   if(cameraController.m_lookMode) {
       //horizontal
       cameraController.m_transform = glm::rotate(cameraController.m_transform, xLook, glm::vec3(0.0f, -1.0f, 0.0f));
       
       //vertical
       cameraController.m_transform = glm::rotate(cameraController.m_transform, yLook, glm::vec3(-1.0f, 0.0f, 0.0f));
   }
   else { //eueler mode
       //horizontal
       cameraController.m_eulerAngles.y -= xLook;
       
       //vertical
       cameraController.m_eulerAngles.x -= yLook;

       if(cameraController.m_eulerAngles.x > 90) {
           cameraController.m_eulerAngles.x = 90;
       }

       if(cameraController.m_eulerAngles.x < -90) {
           cameraController.m_eulerAngles.x = -90;
       }
   }
}

// No mouse event handlers defined
void mouse(int button, int state, int x, int y)
{
   
}

void motion(int x, int y)
{
   mouseLook(x, y);
}
