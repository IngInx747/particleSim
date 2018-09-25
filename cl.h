#ifndef CL_WRAPPER_H
#define CL_WRAPPER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <OpenGL/OpenGL.h> // OpenCL-OpenGL interop
#include <OpenCL/opencl.h>

/** OpenCL C++ Wrapper */
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <CL/cl.hpp>

void pickPlarform(cl::Platform& platform, const std::vector<cl::Platform>& platforms);
void pickDevice(cl::Device& device, const std::vector<cl::Device>& devices);
void printErrorLog(const cl::Program& program, const cl::Device& device);

void initOpenCL(
	cl::CommandQueue & queue,
	cl::Device & device,
	cl::Context & context,
	cl::Program & program,
	const char* source_filename);

#endif