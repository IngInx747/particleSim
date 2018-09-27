#ifndef CL_WRAPPER_H
#define CL_WRAPPER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

#include <OpenGL/OpenGL.h> // OpenCL-OpenGL interop
#include <OpenCL/opencl.h>

/** OpenCL C++ Wrapper */
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include <CL/cl.hpp>

struct CLInfo
{
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
};

void pickPlarform(cl::Platform& platform, const std::vector<cl::Platform>& platforms);
void pickDevice(cl::Device& device, const std::vector<cl::Device>& devices);
void printErrorLog(const cl::Program& program, const cl::Device& device);
//void buildProgram(Program & program, const char* source_filename);
void buildKernel(CLInfo & clInfo, const char* source_filename, const char* func_entry_name, cl::Kernel & kernel);

void initOpenCL(
	cl::Device & device,
	cl::Context & context,
	cl::CommandQueue & queue);

#endif