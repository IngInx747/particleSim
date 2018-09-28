#include "cl.h"

/** Namespace */
using namespace cl;
using namespace std;

//-----------------------------------------------------------------------------
// Initialize OpenCL
//-----------------------------------------------------------------------------

void initOpenCL(
	Device & device,
	Context & context,
	CommandQueue & queue) {

	// Get all available OpenCL platforms
	vector<Platform> platforms;
	Platform::get(&platforms);
	cout << "Available OpenCL platforms :\n\n";
	for (int i = 0; i < platforms.size(); i++)
		cout << "\t" << i+1 << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";

	// Pick a platform
	Platform platform;
	pickPlarform(platform, platforms);
	cout << "\nUsing OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// Get all available OpenCL devices on this platform
	vector<Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	cout << "Available OpenCL devices on this platform :\n\n";
	for (int i = 0; i < devices.size(); i++) {
		cout << "\t" << i+1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
		cout << "\t\tMax compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
		cout << "\t\tMax work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
	}

	// Pick a device
	pickDevice(device, devices);
	cout << "\nUsing OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << "\n";
	cout << "\tMax compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
	cout << "\tMax work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";

	// Create an OpenCL context on that device.
	// Windows specific OpenCL-OpenGL interop
#if defined(_WIN32)
	// Windows                                                                  
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
		0};
#elif defined(__APPLE__)
	// OS X
	#pragma OPENCL EXTENSION CL_APPLE_gl_sharing : enable                                                       
	CGLContextObj     kCGLContext     = CGLGetCurrentContext();
	CGLShareGroupObj  kCGLShareGroup  = CGLGetShareGroup(kCGLContext);
	cl_context_properties properties[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		(cl_context_properties) kCGLShareGroup,
		0};
#else
	// Linux                                                                    
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
		0};
#endif

	// Create an OpenCL context and command queue on the device
	context = Context(device, properties);
	queue = CommandQueue(context, device);
}

//-----------------------------------------------------------------------------
// Compile program
//-----------------------------------------------------------------------------

void buildProgram(CLInfo & clInfo, const char* source_filename, Program & program)
{
	// Convert the OpenCL source code to a string
	ifstream source_file(source_filename, std::ios::in);
	if (!source_file) { cerr << "Cannot find kernel code: " << source_filename << "\n"; exit(1); }
	const string source_string(static_cast<stringstream const&>(stringstream()<<source_file.rdbuf()).str());
	const char* kernel_source = source_string.c_str();

	// Create an OpenCL program by performing runtime compilation for the chosen device
	program = Program(clInfo.context, kernel_source);
	cl_int result = program.build( { clInfo.device } );
	if (result) cout << "Error during compilation OpenCL code!\n (" << result << ")\n";
	if (result == CL_BUILD_PROGRAM_FAILURE) { printErrorLog(program, clInfo.device); exit(1); }
}

//-----------------------------------------------------------------------------
// Compile program & Add program to kernel
//-----------------------------------------------------------------------------

void buildKernel(CLInfo & clInfo, const char* source_filename, const char* func_entry_name, Kernel & kernel)
{
	Program program;
	buildProgram(clInfo, source_filename, program);

	// Add program to kernel
	kernel = cl::Kernel(program, func_entry_name);
}

//-----------------------------------------------------------------------------
// Detect and select a platform as host in OpenCL
//-----------------------------------------------------------------------------

void pickPlarform(Platform& platform, const vector<Platform>& platforms) {

	if (platforms.size() == 1) platform = platforms[0];

	else {
		int input = 0;
		cout << "\nChoose an OpenCL platform: ";
		cin >> input;

		while (input < 1 || input > platforms.size()) {
			cin.clear();
			cin.ignore(cin.rdbuf()->in_avail(), '\n');
			cout << "No such option. Choose an OpenCL platform: ";
			cin >> input;
		}

		platform = platforms[input - 1];
	}
}

//-----------------------------------------------------------------------------
// Detect and select a device for host in OpenCL
//-----------------------------------------------------------------------------

void pickDevice(Device& device, const vector<Device>& devices) {

	if (devices.size() == 1) device = devices[0];

	else {
		int input = 0;
		cout << "\nChoose an OpenCL device: ";
		cin >> input;

		while (input < 1 || input > devices.size()) {
			cin.clear();
			cin.ignore(cin.rdbuf()->in_avail(), '\n');
			cout << "No such option. Choose an OpenCL device: ";
			cin >> input;
		}

		device = devices[input - 1];
	}
}

void printErrorLog(const Program& program, const Device& device) {

	string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	cerr << "Build log:\n" << buildlog << "\n";
}
