#include <iostream>
#include <sstream>
#include <string>

/** Basic GLFW header */
#include <glad/glad.h> // Important - this header must come before glfw3 header
#include <GLFW/glfw3.h>

/** GLFW Math */
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

/** GLFW Texture header */
#include <stb_image/stb_image.h> // Support several formats of image file

/** OpenCL wrapper */
#include <OpenGL/OpenGL.h>
#include "cl.h"

/** Shader Wrapper */
#include <Shader.h>

/** Camera Wrapper */
#include <EularCamera.h>

/** Model Wrapper */
#include <Model.h>

// Particle
struct Particle
{
	glm::vec3 position;
	glm::vec3 velocity;
} __attribute__ ((aligned (16)));

struct ParticleInst
{
	glm::vec4 color;
	glm::mat4 matrix;
};

// Function prototypes
void processInput(GLFWwindow* window);
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void glfw_onFramebufferSize(GLFWwindow* window, int width, int height);
void showFPS(GLFWwindow* window);
bool initOpenGL();

void initKernel(const char* kernel_func_name);
void runKernel();
