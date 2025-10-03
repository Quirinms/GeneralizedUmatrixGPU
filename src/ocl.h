#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
//#include <OpenCL/cl.hpp>
//else
//#include <CL/cl.h>
//#include <CL/cl.hpp>
#elif _WIN32 || _WIN64
#include <CL/cl.h>
//#include <CL/cl.hpp>
#elif __linux__
#include <CL/cl.h>
//#include <CL/cl.hpp>
#endif
