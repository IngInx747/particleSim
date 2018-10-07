#include "main.h"

// Global Variables
const char* APP_TITLE = "sim particle";
const int gWindowWidth = 1280;
const int gWindowHeight = 720;
GLFWwindow* gWindow = NULL;

const unsigned int cnt_obj = 1200; // particles number (<= 1365)
const unsigned int cnt_cell = 1000;

/** OpenCL Global */
cl::Program program;
cl::Kernel kernels[6];
CLInfo clInfo;
cl_int err;

Particle cpuParticles[cnt_obj];
cl::Buffer clParticles;

int cpuIndices[cnt_obj];
cl::Buffer clIndices;

CellLookupTable cpuLookup[cnt_cell];
cl::Buffer clLookup;

cl::Buffer clLambdas;

//std::vector<cl::Memory> clTasks;

// Camera
Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));

//-----------------------------------------------------------------------------
// Main Application Entry Point
//-----------------------------------------------------------------------------

int main() {

	// Init OpenGL
	if (!initOpenGL()){
		// An error occured
		std::cerr << "GLFW initialization failed" << std::endl;
		return -1;
	}
	srand(glfwGetTime());



	// Init OpenCL
	glFinish();
	initOpenCL(clInfo.device, clInfo.context, clInfo.queue);

	// Create buffer for GPU
	clParticles = cl::Buffer(clInfo.context, CL_MEM_READ_WRITE, cnt_obj * sizeof(Particle));
	clIndices = cl::Buffer(clInfo.context, CL_MEM_READ_WRITE, cnt_obj * sizeof(int));
	clLookup = cl::Buffer(clInfo.context, CL_MEM_READ_WRITE, cnt_cell * sizeof(CellLookupTable));
	clLambdas = cl::Buffer(clInfo.context, CL_MEM_READ_WRITE, cnt_obj * sizeof(float));

	// Create program for kernels
	buildProgram(clInfo, "Particle.cl", program);

	// Specify OpenCL kernel arguments (args[0] here is entry function name of GPU)
	buildKernel(clInfo, program, "kernel_externel_force", kernels[0]);
	buildKernel(clInfo, program, "kernel_find_cell", kernels[1]);
	buildKernel(clInfo, program, "kernel_calc_lambda", kernels[2]);
	buildKernel(clInfo, program, "kernel_calc_disp", kernels[3]);
	buildKernel(clInfo, program, "kernel_update", kernels[4]);
	buildKernel(clInfo, program, "kernel_viscosity", kernels[5]);



	// Model loader
	Model objectParticle("Resources/sphere/sphere.obj");

	// Shader loader
	Shader objectShader, instanceShader;
	objectShader.loadShaders("shaders/demo.vert", "shaders/demo.frag");
	instanceShader.loadShaders("shaders/instancing.vert", "shaders/instancing.frag");



	// Light global
	glm::vec3 pointLightPos[] = {
		glm::vec3( 3.0f,  0.0f,  0.0f),
		glm::vec3(-3.0f,  0.0f,  0.0f),
		glm::vec3( 0.0f,  0.0f, -3.0f),
		glm::vec3( 0.0f,  0.0f,  3.0f)
	};
	glm::vec3 directionalLightDirection(0.0f, 0.0f, -1.0f);

	// Object shader config
	objectShader.use();
	// Light config
	objectShader.setUniform("uDirectionalLight.direction", directionalLightDirection);
	objectShader.setUniform("uDirectionalLight.ambient", 0.5f, 0.5f, 0.5f);
	objectShader.setUniform("uDirectionalLight.diffuse", 1.0f, 1.0f, 1.0f);
	objectShader.setUniform("uDirectionalLight.specular", 1.0f, 1.0f, 1.0f);
	objectShader.setUniform("uSpotLight.innerCutOff", glm::cos(glm::radians(12.5f)));
	objectShader.setUniform("uSpotLight.outerCutOff", glm::cos(glm::radians(17.5f)));
	objectShader.setUniform("uSpotLight.ambient", 0.0f, 0.0f, 0.0f);
	objectShader.setUniform("uSpotLight.diffuse", 1.0f, 1.0f, 1.0f);
	objectShader.setUniform("uSpotLight.specular", 1.0f, 1.0f, 1.0f);
	objectShader.setUniform("uSpotLight.constant", 1.0f);
	objectShader.setUniform("uSpotLight.linear", 0.09f);
	objectShader.setUniform("uSpotLight.quadratic", 0.032f);

	instanceShader.use();
	instanceShader.setUniform("uDirectionalLight.direction", directionalLightDirection);
	instanceShader.setUniform("uDirectionalLight.ambient", 0.0f, 0.0f, 0.0f);
	instanceShader.setUniform("uDirectionalLight.diffuse", 1.0f, 1.0f, 1.0f);
	instanceShader.setUniform("uDirectionalLight.specular", 1.0f, 1.0f, 1.0f);
	instanceShader.setUniform("uSpotLight.innerCutOff", glm::cos(glm::radians(12.5f)));
	instanceShader.setUniform("uSpotLight.outerCutOff", glm::cos(glm::radians(17.5f)));
	instanceShader.setUniform("uSpotLight.ambient", 0.0f, 0.0f, 0.0f);
	instanceShader.setUniform("uSpotLight.diffuse", 1.0f, 1.0f, 1.0f);
	instanceShader.setUniform("uSpotLight.specular", 1.0f, 1.0f, 1.0f);
	instanceShader.setUniform("uSpotLight.constant", 1.0f);
	instanceShader.setUniform("uSpotLight.linear", 0.09f);
	instanceShader.setUniform("uSpotLight.quadratic", 0.032f);



	// Instancing
	ParticleInst particleInst[cnt_obj];

	unsigned int ibo;
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ARRAY_BUFFER, ibo);

	for (Mesh & mesh : objectParticle.meshes) {

		unsigned VAO = mesh.VAO();
		glBindVertexArray(VAO);
		size_t vec4Size = (int) sizeof(glm::vec4);

		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleInst), (void*)0);

		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleInst), (void*)(1 * vec4Size));
		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleInst), (void*)(2 * vec4Size));
		glEnableVertexAttribArray(6);
		glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleInst), (void*)(3 * vec4Size));
		glEnableVertexAttribArray(7);
		glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleInst), (void*)(4 * vec4Size));

		glVertexAttribDivisor(3, 1);
		glVertexAttribDivisor(4, 1);
		glVertexAttribDivisor(5, 1);
		glVertexAttribDivisor(6, 1);
		glVertexAttribDivisor(7, 1);

		glBindVertexArray(0);
	}



	// Init Particle
	for (int i = 0; i < cnt_obj; i++) {
		//float px = (rand() / (float) RAND_MAX) * 2.0f - 1.0f;
		//float py = 0.0f;
		//float pz = (rand() / (float) RAND_MAX) * 2.0f - 1.0f;
		//float vx = (rand() / (float) RAND_MAX) * 2.0f - 1.0f;
		//float vy = (rand() / (float) RAND_MAX) * 2.0f - 1.0f;
		//float vz = (rand() / (float) RAND_MAX) * 2.0f - 1.0f;
		float px = -0.9f + 0.2f * (i % 10);
		float py = -0.9f + 0.2f * (i / 100);
		float pz = -0.9f + 0.2f * ((i / 10) % 10);
		float vx = 0.0f;
		float vy = 0.0f;
		float vz = 0.0f;
		cpuParticles[i].position = glm::vec3(px, py, pz);
		cpuParticles[i].velocity = glm::vec3(vx, vy, vz);
	}

	clInfo.queue.enqueueWriteBuffer(clParticles, CL_TRUE, 0, cnt_obj * sizeof(Particle), cpuParticles);

	//
	kernels[0].setArg(0, clParticles);
	//
	kernels[1].setArg(0, clParticles);
	kernels[1].setArg(1, clIndices);
	//
	kernels[2].setArg(0, clParticles);
	kernels[2].setArg(1, clLookup);
	kernels[2].setArg(2, clIndices);
	kernels[2].setArg(3, clLambdas);
	//
	kernels[3].setArg(0, clParticles);
	kernels[3].setArg(1, clLookup);
	kernels[3].setArg(2, clIndices);
	kernels[3].setArg(3, clLambdas);
	//
	kernels[4].setArg(0, clParticles);
	//
	//kernels[5].setArg(0, clParticles);
	//kernels[5].setArg(1, clLookup);
	//kernels[5].setArg(2, clIndices);

	// Rendering loop
	while (!glfwWindowShouldClose(gWindow)) {

		// Display FPS on title
		showFPS(gWindow);

		// Key input
		processInput(gWindow);

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



		////////// Fluid calculation //////////

		// apply external force
		runKernel(kernels[0], clInfo);

		// find particles' neighbors
		runKernel(kernels[1], clInfo);

		clInfo.queue.enqueueReadBuffer(clIndices, CL_TRUE, 0, cnt_obj * sizeof(int), cpuIndices);

		std::map<int, std::vector<int> > cell_particle_table; // not radix-sorting yet
		for (int i = 0; i < cnt_obj; i++)
		{
			//std::cout<<i<<" "<<cpuIndices[i]<<"\n";
			cell_particle_table[cpuIndices[i]].push_back(i);
		}

		int cnt_particle = 0;
		for (auto iter = cell_particle_table.begin(); iter != cell_particle_table.end(); iter++)
		{
			//std::cout<<iter->first<<" "<<cnt_particle<<" "<<iter->second.size()<<"\n";
			cpuLookup[iter->first].offset = cnt_particle;
			cpuLookup[iter->first].size = iter->second.size(); // number of particles in current cell

			for (int p_id : iter->second)
			{
				cpuIndices[cnt_particle++] = p_id; // particle ID
			}
		}

		//for (int i=0; i<cnt_cell; i++)
		//{
		//	if (cpuLookup[i].size == 0) continue;
		//	std::cout<<i<<" "<<cpuLookup[i].offset<<" "<<cpuLookup[i].size<<" : ";
		//	for (int j=0; j<cpuLookup[i].size; j++)
		//	{
		//		std::cout<<cpuIndices[cpuLookup[i].offset + j]<<" ";
		//	} std::cout<<"\n";
		//}

		clInfo.queue.enqueueWriteBuffer(clIndices, CL_TRUE, 0, cnt_obj * sizeof(int), cpuIndices);
		clInfo.queue.enqueueWriteBuffer(clLookup, CL_TRUE, 0, cnt_cell * sizeof(CellLookupTable), cpuLookup);

		// solve constrain equation
		unsigned int num_iteration = 5;
		for (int i = 0; i < num_iteration; ++i)
		{
			// calculate lambda
			runKernel(kernels[2], clInfo);

			// calculate displacement
			runKernel(kernels[3], clInfo);
		}

		// update particle
		runKernel(kernels[4], clInfo);

		// confining fluid
		//runKernel(kernels[5], clInfo);

		clInfo.queue.enqueueReadBuffer(clParticles, CL_TRUE, 0, cnt_obj * sizeof(Particle), cpuParticles);



		for (unsigned int i=0; i<cnt_obj; i++) {

			// transformation

			glm::mat4 matrix;

			float rx = cpuParticles[i].position.x;
			float ry = cpuParticles[i].position.y;
			float rz = cpuParticles[i].position.z;

			matrix = glm::translate(matrix, glm::vec3(rx, ry, rz));
			matrix = glm::scale(matrix, glm::vec3(0.02f));

			particleInst[i].matrix = matrix;

			// speed discriminator

			float speed = glm::length(cpuParticles[i].velocity);
			speed = 1.0f - std::exp(-speed);
			particleInst[i].color = glm::vec4(speed, speed, 1.0f, 1.0f);
		}

		glBufferData(GL_ARRAY_BUFFER, cnt_obj * sizeof(ParticleInst), &particleInst[0], GL_STATIC_DRAW);



		// Camera transformations
		glm::mat4 view = camera.getViewMatrix();
		float width_height_ratio = (float)gWindowWidth / (float)gWindowHeight;
		glm::mat4 projection = glm::perspective(glm::radians(camera.fov), width_height_ratio, 0.1f, 1000.0f);

		objectShader.use();
		objectShader.setUniform("uView", view);
		objectShader.setUniform("uProjection", projection);
		objectShader.setUniform("uCameraPos", camera.position);
		objectShader.setUniform("uSpotLight.position", camera.position);
		objectShader.setUniform("uSpotLight.direction", camera.front);

		instanceShader.use();
		instanceShader.setUniform("uView", view);
		instanceShader.setUniform("uProjection", projection);
		instanceShader.setUniform("uCameraPos", camera.position);
		instanceShader.setUniform("uSpotLight.position", camera.position);
		instanceShader.setUniform("uSpotLight.direction", camera.front);



		// Draw Models
		//glm::mat4 modelMatrix;
		//modelMatrix = glm::mat4(1.0f);
		//modelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, -3.0f, 0.0f));
		//modelMatrix = glm::scale(modelMatrix, glm::vec3(4.0f, 4.0f, 4.0f));
		//objectShader.use();
		//objectShader.setUniform("uModel", modelMatrix);
		//objectParticle.Draw(objectParticle);

		instanceShader.use();
		instanceShader.setUniform("uMaterial.texture_diffuse1", 0);
		//glActiveTexture(GL_TEXTURE0);
		//glBindTexture(GL_TEXTURE_2D, objectParticle.textures_loaded[0].id);
		for (Mesh & mesh : objectParticle.meshes) {
			glBindVertexArray(mesh.VAO());
			glDrawElementsInstanced(GL_TRIANGLES, mesh.indices.size(), GL_UNSIGNED_INT, 0, cnt_obj);
			glBindVertexArray(0);
		}



		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwPollEvents();
		glfwSwapBuffers(gWindow);
	}
	
	glfwTerminate();

	return 0;
}



//-----------------------------------------------------------------------------
// execute kernel
//-----------------------------------------------------------------------------

void runKernel(cl::Kernel & kernel, CLInfo & clInfo) {

	// Every pixel in the image has its own thread or "work item",
	// so #work_items == #pixel
	std::size_t global_work_size = cnt_obj;
	std::size_t local_work_size = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(clInfo.device);
	// Ensure the global work size is a multiple of local work size
	if (global_work_size % local_work_size != 0)
		global_work_size = (global_work_size / local_work_size + 1) * local_work_size;

	glFinish();
	clInfo.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);
	clInfo.queue.finish();
}



//-----------------------------------------------------------------------------
// Initialize GLFW and OpenGL
//-----------------------------------------------------------------------------
bool initOpenGL() {

	// Intialize GLFW 
	// GLFW is configured.  Must be called before calling any GLFW functions
	if (!glfwInit()) {
		// An error occured
		std::cerr << "GLFW initialization failed" << std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// forward compatible with newer versions of OpenGL as they become available
	// but not backward compatible (it will not run on devices that do not support OpenGL 3.3
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// Create an OpenGL 3.3 core, forward compatible context window
	gWindow = glfwCreateWindow(gWindowWidth, gWindowHeight, APP_TITLE, NULL, NULL);
	if (gWindow == NULL) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return false;
	}

	// Make the window's context the current one
	glfwMakeContextCurrent(gWindow);

	// Set the required callback functions
	//glfwSetKeyCallback(gWindow, glfw_onKey);
	glfwSetCursorPosCallback(gWindow, mouseCallback);
	glfwSetScrollCallback(gWindow, scrollCallback);
	glfwSetFramebufferSizeCallback(gWindow, glfw_onFramebufferSize);

	// Initialize GLAD: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		return false;
	}

	glClearColor(0.3f, 0.3f, 0.3f, 1.0f);

	// Define the viewport dimensions
	//glViewport(0, 0, gWindowWidth, gWindowHeight);

	// Configure global OpenGL state	
	glEnable(GL_DEPTH_TEST);

	// Blending functionality
	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Hide the cursor and capture it
	glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	return true;
}

//-----------------------------------------------------------------------------
// Is called whenever a key is pressed/released via GLFW
//-----------------------------------------------------------------------------
void processInput(GLFWwindow* window) {

	// FPS
	static float deltaTime = 0.0f;
	static float lastFrame = 0.0f;

	// Per-frame time
	float currentFrame = (float) glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;

	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		camera.processAccerlate(true);
	else
		camera.processAccerlate(false);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.processKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.processKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.processKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.processKeyboard(RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		camera.processKeyboard(UP, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		camera.processKeyboard(DOWN, deltaTime);
	
	static bool gWireframe = false;
	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
		gWireframe = !gWireframe;
		if (gWireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

//-----------------------------------------------------------------------------
// Is called whenever mouse movement is detected via GLFW
//-----------------------------------------------------------------------------
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {

	static bool firstMouse = true;
	static float lastX = gWindowWidth / 2;
	static float lastY = gWindowHeight / 2;

	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coord range from buttom to top
	lastX = xpos;
	lastY = ypos;

	camera.processMouse(xoffset, yoffset);
}

//-----------------------------------------------------------------------------
// Is called whenever scroller is detected via GLFW
//-----------------------------------------------------------------------------
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.processScroll(yoffset);
}

//-----------------------------------------------------------------------------
// Is called when the window is resized
//-----------------------------------------------------------------------------
void glfw_onFramebufferSize(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

//-----------------------------------------------------------------------------
// Code computes the average frames per second, and also the average time it takes
// to render one frame.  These stats are appended to the window caption bar.
//-----------------------------------------------------------------------------
void showFPS(GLFWwindow* window)
{
	static double previousSeconds = 0.0;
	static int frameCount = 0;
	double elapsedSeconds;
	double currentSeconds = glfwGetTime(); // returns number of seconds since GLFW started, as double float

	elapsedSeconds = currentSeconds - previousSeconds;

	// Limit text updates to 4 times per second
	if (elapsedSeconds > 0.25)
	{
		previousSeconds = currentSeconds;
		double fps = (double)frameCount / elapsedSeconds;
		double msPerFrame = 1000.0 / fps;

		// The C++ way of setting the window title
		std::ostringstream outs;
		outs.precision(3);	// decimal places
		outs << std::fixed
			<< APP_TITLE << "    "
			<< "FPS: " << fps << "    "
			<< "Frame Time: " << msPerFrame << " (ms)";
		glfwSetWindowTitle(window, outs.str().c_str());

		// Reset for next average.
		frameCount = 0;
	}

	frameCount++;
}
