
#include "ocl.h"
//#include <CL/cl.h>

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace Rcpp;

std::string get_UpdateWeights(){
  
  std::string UpdateWeights = R"(
    __kernel void update_weights(
        __global float* esom,
        __global float* DataSample,
        __global float* OutputDistances,
        const int RowIdx,
        const int N,
        const int Lines,
        const int Columns,
        const int Weights,
        const int Radius,
        const float Factor){
      
      size_t k = get_global_id(0);
      size_t j = get_global_id(1);
      size_t i = get_global_id(2);
      
      if((k < Lines) & (j < Columns) & (i < Weights)){
        float pi            = 3.1416;
        float tmpVar1       = OutputDistances[k + j*Lines] * OutputDistances[k + j*Lines];
        float tmpVar2       = (float) (Radius * Radius);
        float neighborValue = 1.0 - (tmpVar1 / (pi*tmpVar2));
        
        if(neighborValue < 0.0){
          neighborValue = 0.0;
        }
        
        int tmpIdx1   = i * Columns * Lines + j * Lines + k;
        float tmpRes0 = esom[tmpIdx1];
        esom[tmpIdx1] = tmpRes0 - (Factor * (neighborValue * (tmpRes0 - DataSample[RowIdx + i * N])));
      }
    }
  
  )";
  
  return(UpdateWeights);
}

std::string get_ToroidDistance(){
  
  std::string ToroidDistance = R"(
    __kernel void toroid_distance(
        const float bm1,
        const float bm2,
        const float CorrectLines,
        const float CorrectColumns,
        const int Lines,
        const int Columns,
        const int LCS,
        __global float* OutputDistances){
      
      size_t i = get_global_id(0);
      size_t j = get_global_id(1);
      
      if((i < Lines) & (j < Columns)){
        float tmpVar1                = CorrectLines - fabs(2.0 * fabs(((float) i) - bm1) - CorrectLines);
        float FirstPart              = tmpVar1 * tmpVar1;
        float tmpVar2                = CorrectColumns - fabs(2.0 * fabs(((float) j) - bm2) - CorrectColumns);
        float SecondPart             = tmpVar2 * tmpVar2;
        OutputDistances[j*Lines + i] = 0.5f*sqrt(FirstPart + SecondPart);
        
        // Symmetrie ist nicht ausnutzbar: keine quadratische Matrix!
        //OutputDistances(j,i) = OutputDistances(i,j);
      }
    }
  )";
  
  return(ToroidDistance);
}

std::string get_NonToroidDistance(){
  
  std::string NonToroidDistance = R"(
    __kernel void non_toroid_distance(
        const float bm1,
        const float bm2,
        const int Lines,
        const int Columns,
        __global float* OutputDistances){
      
      size_t i = get_global_id(0);
      size_t j = get_global_id(1);
      
      if((i < Lines) & (j < Columns)){
        OutputDistances[j*Lines + i] = sqrt(pow(i - bm1, 2) + pow(j - bm2, 2));
        // Symmetrie ist nicht ausnutzbar: keine quadratische Matrix!
        //OutputDistances(j,i) = OutputDistances(i,j);
      }
    }
  )";
  
  return(NonToroidDistance);
}

// static char* get_info_string(cl_platform_id plat, cl_platform_info param) {
//   size_t size = 0;
//   cl_int err = clGetPlatformInfo(plat, param, 0, NULL, &size);
//   if (err != CL_SUCCESS) return NULL;
//   char *buf = (char*)malloc(size + 1);
//   if (!buf) return NULL;
//   err = clGetPlatformInfo(plat, param, size, buf, NULL);
//   if (err != CL_SUCCESS) { free(buf); return NULL; }
//   buf[size] = '\0';
//   return buf;
// }
// 
// static char* get_device_info_string(cl_device_id dev, cl_device_info param) {
//   size_t size = 0;
//   cl_int err = clGetDeviceInfo(dev, param, 0, NULL, &size);
//   if (err != CL_SUCCESS) return NULL;
//   char *buf = (char*)malloc(size + 1);
//   if (!buf) return NULL;
//   err = clGetDeviceInfo(dev, param, size, buf, NULL);
//   if (err != CL_SUCCESS) { free(buf); return NULL; }
//   buf[size] = '\0';
//   return buf;
// }
// 
// static void print_device_type(cl_device_id dev) {
//   cl_device_type type;
//   if (clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, NULL) != CL_SUCCESS) {
//     printf("    Type: <unknown>\n");
//     return;
//   }
//   printf("    Type: ");
//   if (type & CL_DEVICE_TYPE_CPU) printf("CPU ");
//   if (type & CL_DEVICE_TYPE_GPU) printf("GPU ");
//   if (type & CL_DEVICE_TYPE_ACCELERATOR) printf("ACCELERATOR ");
//   if (type & CL_DEVICE_TYPE_DEFAULT) printf("DEFAULT ");
//   printf("\n");
// }

//static void die(const char *msg, cl_int err) {
//  if (err != CL_SUCCESS) {
//    fprintf(stderr, "%s (err = %d)\n", msg, err);
//    exit(EXIT_FAILURE);
//  } else {
//    fprintf(stderr, "%s\n", msg);
//    exit(EXIT_FAILURE);
//  }
//}

std::vector<float> trainstepC3(std::vector<float> esomwts,
                               std::vector<float> DataSampled,
                               std::vector<float> BMUsampled,
                               std::vector<int> Index,
                               int N, int DIM, int NumDataPerEpoch,
                               int Lines, int Columns, int Weights, int Radius,
                               bool toroid, int Iteration){
  
  // OpenCL device search ------------------------------------------------------
  //cl_int err;
  cl_uint numPlatforms = 0;
  clGetPlatformIDs(0, nullptr, &numPlatforms);
  
  //if(err != CL_SUCCESS) die("Failed to query number of platforms", err);
  
  if(numPlatforms == 0){
    //printf("No OpenCL platforms found.\n");
    std::vector<float> ZeroRes(Lines*Columns*Weights);
    return ZeroRes; // return 0-vector of size Lines*Columns*Weights
  }
  
  std::vector<cl_platform_id> platforms(numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  
  //std::cout << "Number of platforms: " << numPlatforms << "\n";
  
  //cl_uint numDevices = 0;
  //clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  //std::vector<cl_device_id> devices(numDevices);
  //clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
  
  //MAX_COMPUTE_UNITS
  //clGetDeviceInfo(devices[0], VENDOR, numDevices, devices.data(), nullptr);
  //char *plat_vendor = get_info_string(platforms[0], CL_PLATFORM_VENDOR);
  //printf("Vendor: %s\n", plat_vendor ? plat_vendor : "<unknown>");
  //std::cout << "numDevices: " << numDevices << "\n";
  //std::cout << "platforms: " << platforms[0] << "\n";
  //std::cout << "devices: " << devices[0] << "\n";
  
  cl_uint PlatformIdx = 0;
  cl_uint DeviceNumber = 0;
  cl_uint MaxCU = 0;
  
  for(cl_uint i = 0; i < numPlatforms; ++i){
    //printf("  Platform %u:\n", i);
    
    cl_uint numDevicesSearch = 0;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevicesSearch);
    std::vector<cl_device_id> deviceSearch(numDevicesSearch);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevicesSearch, deviceSearch.data(), nullptr);
    
    for(cl_uint d = 0; d < numDevicesSearch; ++d){
      //printf("  Device %u:\n", d);
      
      // char *dev_name = get_device_info_string(deviceSearch[d], CL_DEVICE_NAME);
      // char *dev_vendor = get_device_info_string(deviceSearch[d], CL_DEVICE_VENDOR);
      // char *dev_driver = get_device_info_string(deviceSearch[d], CL_DRIVER_VERSION);
      // char *dev_version = get_device_info_string(deviceSearch[d], CL_DEVICE_VERSION);
      // char *dev_profile = get_device_info_string(deviceSearch[d], CL_DEVICE_PROFILE);
      // char *dev_extensions = get_device_info_string(deviceSearch[d], CL_DEVICE_EXTENSIONS);
      
      //printf("    Name:                    %s\n", dev_name ? dev_name : "<unknown>");
      //printf("    Vendor:                  %s\n", dev_vendor ? dev_vendor : "<unknown>");
      //if (dev_driver) printf("    Driver version:          %s\n", dev_driver);
      //if (dev_version) printf("    Device OpenCL version:   %s\n", dev_version);
      //if (dev_profile) printf("    Device profile:          %s\n", dev_profile);
      //print_device_type(deviceSearch[d]);
      
      cl_uint compute_units = 0;
      cl_ulong global_mem = 0;
      cl_ulong max_alloc = 0;
      cl_ulong local_mem = 0;
      cl_uint max_clock = 0;
      size_t max_wg_size = 0;
      cl_bool image_support = CL_FALSE;
      
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock), &max_clock, NULL);
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg_size), &max_wg_size, NULL);
      clGetDeviceInfo(deviceSearch[d], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
      
      if(compute_units > MaxCU){
        MaxCU        = compute_units;
        PlatformIdx  = i;
        DeviceNumber = d;
      }
      
      //printf("    Max compute units:       %u\n", compute_units);
      //printf("    Max clock (MHz):         %u\n", max_clock);
      //printf("    Global mem size (bytes): %llu\n", (unsigned long long)global_mem);
      //printf("    Max alloc (bytes):       %llu\n", (unsigned long long)max_alloc);
      //printf("    Local mem size (bytes):  %llu\n", (unsigned long long)local_mem);
      //printf("    Max work-group size:     %zu\n", max_wg_size);
      //printf("    Image support:           %s\n", image_support ? "yes" : "no");
    }
    //free(deviceSearch);
  }
  
  // OpenCL device search END --------------------------------------------------
  
  cl_uint numDevices = 0;
  clGetDeviceIDs(platforms[PlatformIdx], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
  std::vector<cl_device_id> devices(numDevices);
  clGetDeviceIDs(platforms[PlatformIdx], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
  
  //char *dev_name = get_device_info_string(devices[DeviceNumber], CL_DEVICE_NAME);
  //printf("Name: %s\n", dev_name ? dev_name : "<unknown>");
  
  int LCS = Lines * Columns;
  float Factor = 1.0;
  
  //std::vector<float> DataRow(DIM);
  
  if((N >= 2501) && (Radius <= 16)){
    if(Radius <= 16 && Radius > 8){
      Factor = 0.75;
    }else if (Radius <= 8 && Radius > 4){
      Factor = 0.5;
    }else{
      Factor = 0.1;
    }
  }
  
  cl_int Error0, Error1, Error2, Error3;
  // Build and compile the kernel
  cl_context MyOCLContext     = clCreateContext(nullptr, 1, &devices[DeviceNumber], nullptr, nullptr, &Error0);
  cl_command_queue MyOCLQueue = clCreateCommandQueue(MyOCLContext, devices[DeviceNumber], 0, &Error0);
  //cl_command_queue MyOCLQueue = clCreateCommandQueueWithProperties(MyOCLContext, devices[DeviceNumber], 0, &Error0);
  
  //if(toroid == true){
    std::string KernelToroidDistance = get_ToroidDistance();
    const char* Code1 = KernelToroidDistance.c_str();
    size_t CodeLength1 = KernelToroidDistance.size();
    cl_program Program_ToroidDistance = clCreateProgramWithSource(MyOCLContext, 1, &Code1, &CodeLength1, &Error1);
    clBuildProgram(Program_ToroidDistance, 1, &devices[DeviceNumber], nullptr, nullptr, nullptr);
    cl_kernel Kernel_ToroidDistance = clCreateKernel(Program_ToroidDistance, "toroid_distance", &Error1);
    size_t SizeToroidDistance[2] = {(size_t) Lines, (size_t) Columns};
  //}else{
    std::string KernelNonToroidDistance = get_NonToroidDistance();
    const char* Code2 = KernelNonToroidDistance.c_str();
    size_t CodeLength2 = KernelNonToroidDistance.size();
    cl_program Program_NonToroidDistance = clCreateProgramWithSource(MyOCLContext, 1, &Code2, &CodeLength2, &Error2);
    clBuildProgram(Program_NonToroidDistance, 1, &devices[DeviceNumber], nullptr, nullptr, nullptr);
    cl_kernel Kernel_NonToroidDistance = clCreateKernel(Program_NonToroidDistance, "non_toroid_distance", &Error2);
    size_t SizeNonToroidDistance[2] = {(size_t) Lines, (size_t) Columns};
  //}
  
  std::string KernelUpdateWeights = get_UpdateWeights();
  const char* Code3 = KernelUpdateWeights.c_str();
  size_t CodeLength3 = KernelUpdateWeights.size();
  cl_program Program_UpdateWeights = clCreateProgramWithSource(MyOCLContext, 1, &Code3, &CodeLength3, &Error3);
  clBuildProgram(Program_UpdateWeights, 1, &devices[DeviceNumber], nullptr, nullptr, nullptr);
  cl_kernel Kernel_UpdateWeights = clCreateKernel(Program_UpdateWeights, "update_weights", &Error3);
  size_t SizeUpdateWeights[3] = {(size_t) Lines, (size_t) Columns, (size_t) Weights};
  
  // Initiate/Instantiate OpenCL buffers:
  // Create OpenCL buffers for static/const variables
  // CPP API for OpenCL:
  //cl::Buffer clData(MyOCLContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * DIM, DataSampled.data());
  // C API for OpenCL:
  cl_mem clData = clCreateBuffer(MyOCLContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * DIM, DataSampled.data(), &Error0);
  // Create OpenCL buffers for changing variables
  cl_mem clESOM = clCreateBuffer(MyOCLContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * Lines*Columns*Weights, esomwts.data(), &Error0);
  cl_mem clDM   = clCreateBuffer(MyOCLContext, CL_MEM_READ_WRITE, sizeof(float) * Lines*Columns, nullptr, &Error0);
  
  //cl::Buffer clESOM(MyOCLContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * Lines*Columns*Weights, esomwts.data());
  //cl::Buffer clDM(MyOCLContext, CL_MEM_READ_WRITE, sizeof(float) * Lines*Columns);
  
  float AdjustedLines   = (float) Lines - 1;
  float AdjustedColumns = (float) Columns - 1;
  
  //for(int p = 0; p < N; p++){
  for(int p = 0; p < NumDataPerEpoch; p++){
    
    int DataIdx = Index[p];
    
    float bmpos0 = BMUsampled[DataIdx];
    float bmpos1 = BMUsampled[DataIdx + N];
    
    if(toroid == true){
      // Set kernel arguments
      clSetKernelArg(Kernel_ToroidDistance, 0, sizeof(float), &bmpos0);
      clSetKernelArg(Kernel_ToroidDistance, 1, sizeof(float), &bmpos1);
      clSetKernelArg(Kernel_ToroidDistance, 2, sizeof(float), &AdjustedLines);
      clSetKernelArg(Kernel_ToroidDistance, 3, sizeof(float), &AdjustedColumns);
      clSetKernelArg(Kernel_ToroidDistance, 4, sizeof(int), &Lines);
      clSetKernelArg(Kernel_ToroidDistance, 5, sizeof(int), &Columns);
      clSetKernelArg(Kernel_ToroidDistance, 6, sizeof(int), &LCS);
      clSetKernelArg(Kernel_ToroidDistance, 7, sizeof(cl_mem), &clDM); //  * Lines*Columns
      clEnqueueNDRangeKernel(MyOCLQueue, Kernel_ToroidDistance, 2, nullptr, SizeToroidDistance, nullptr, 0, nullptr, nullptr);
    }else{
      // Set kernel arguments
      clSetKernelArg(Kernel_NonToroidDistance, 0, sizeof(float), &bmpos0);
      clSetKernelArg(Kernel_NonToroidDistance, 1, sizeof(float), &bmpos1);
      clSetKernelArg(Kernel_NonToroidDistance, 2, sizeof(int), &Lines);
      clSetKernelArg(Kernel_NonToroidDistance, 3, sizeof(int), &Columns);
      clSetKernelArg(Kernel_NonToroidDistance, 4, sizeof(cl_mem), &clDM); //  * Lines*Columns
      clEnqueueNDRangeKernel(MyOCLQueue, Kernel_NonToroidDistance, 2, nullptr, SizeNonToroidDistance, nullptr, 0, nullptr, nullptr);
    }
    
    // Set kernel arguments
    clSetKernelArg(Kernel_UpdateWeights, 0, sizeof(cl_mem), &clESOM); // * Lines*Columns*Weights
    clSetKernelArg(Kernel_UpdateWeights, 1, sizeof(cl_mem), &clData);// * N*DIM
    clSetKernelArg(Kernel_UpdateWeights, 2, sizeof(cl_mem), &clDM);// * Lines*Columns
    clSetKernelArg(Kernel_UpdateWeights, 3, sizeof(int), &DataIdx);
    clSetKernelArg(Kernel_UpdateWeights, 4, sizeof(int), &N);
    clSetKernelArg(Kernel_UpdateWeights, 5, sizeof(int), &Lines);
    clSetKernelArg(Kernel_UpdateWeights, 6, sizeof(int), &Columns);
    clSetKernelArg(Kernel_UpdateWeights, 7, sizeof(int), &Weights);
    clSetKernelArg(Kernel_UpdateWeights, 8, sizeof(int), &Radius);
    clSetKernelArg(Kernel_UpdateWeights, 9, sizeof(float), &Factor);
    clEnqueueNDRangeKernel(MyOCLQueue, Kernel_UpdateWeights, 3, nullptr, SizeUpdateWeights, nullptr, 0, nullptr, nullptr);
  }
  
  //MyOCLQueue.finish();
  //MyOCLQueue.enqueueReadBuffer(clESOM, CL_TRUE, 0, sizeof(float) * Lines*Columns*Weights, esomwts.data());
  clEnqueueReadBuffer(MyOCLQueue, clESOM, CL_TRUE, 0, sizeof(float) * Lines*Columns*Weights, esomwts.data(), 0, NULL, NULL);
  
  clReleaseKernel(Kernel_ToroidDistance);
  clReleaseKernel(Kernel_NonToroidDistance);
  clReleaseKernel(Kernel_UpdateWeights);
  
  clReleaseProgram(Program_ToroidDistance);
  clReleaseProgram(Program_NonToroidDistance);
  clReleaseProgram(Program_UpdateWeights);
  
  clReleaseMemObject(clESOM);
  clReleaseMemObject(clData);
  clReleaseMemObject(clDM);
  
  clReleaseCommandQueue(MyOCLQueue);
  clReleaseContext(MyOCLContext);
  
  //free(platforms);
  
  return(esomwts);
}

// [[Rcpp::export]]
NumericVector trainSESOM(NumericVector Data, NumericVector BMUs, NumericVector RadiusVector,
                         int N, int DIM, double MinData, double MaxData,
                         int Lines, int Columns, int Weights,
                         bool toroid, int NumDataPerEpoch){
  
  int CurrentRadius;
  
  int sizeArea  = Lines*Columns;     // Lines*Columns
  int sizeESOM  = sizeArea*Weights;  // Lines*Columns*Weights
  int NumEpochs = RadiusVector.length();
  
  std::vector<float> DataVector(N * DIM);
  std::copy(Data.begin(), Data.end(), DataVector.begin());
  std::vector<float> BMUvector(2 * N);
  std::copy(BMUs.begin(), BMUs.end(), BMUvector.begin());
  
  // Random device and generator
  std::random_device rd;
  std::mt19937 gen(rd());
  
  std::vector<float> esomwts;
  
  // Uniform distribution on [MinData, MaxData)
  std::uniform_real_distribution<float> dist((float) MinData, (float) MaxData);
  
  // Generate n samples
  for(int i = 0; i < sizeESOM; i++){
    esomwts.push_back(dist(gen));
  }
  
  std::vector<int> BatchIndex;
  std::vector<int> KeyBot(N);
  
  for(int i = 0; i < N; i++){
    KeyBot[i] = i;
  }
  
  //float progress = 0.0;
  
  for(int i = 0; i < NumEpochs; i++){  // Train ESOM with decreasing radius
    
    if(N > NumDataPerEpoch){
      std::random_device rd2;
      std::mt19937 gen2(rd2());
      std::sample(KeyBot.begin(), KeyBot.end(), std::back_inserter(BatchIndex), NumDataPerEpoch, gen2);
    }
    
    //std::cout << "#---------------#" << "\n";
    //std::cout << "Epoch: " << i << "\n";
    //std::cout << "#---------------#" << "\n";
    
    CurrentRadius = RadiusVector[i];
    //toroid = 1;
    
    // Perform permutation to shuffle the order of data!
    // std::vector<int> KeyBot(N);
    // for(int j = 0; j < N; j++){
    //   KeyBot[j] = j;
    // }
    // std::vector<int> KeyBot2 = KeyBot;
    // std::random_device rd2;
    // std::mt19937 gen2(rd2());
    // std::shuffle(KeyBot2.begin(), KeyBot2.end(), gen2);
    // for(int j = 0; j < N; j++){
    //   int Row1 = KeyBot[j];
    //   int Row2 = KeyBot2[j];
    //   
    //   for(int k = 0; k < DIM; k++){
    //     double tmpSwap1          = DataVector[Row1 + k * N];
    //     DataVector[Row1 + k * N] = DataVector[Row2 + k * N];
    //     DataVector[Row2 + k * N] = tmpSwap1;
    //   }
    //   
    //   for(int k = 0; k < 2; k++){
    //     double tmpSwap2         = BMUvector[Row1 + k * N];
    //     BMUvector[Row1 + k * N] = BMUvector[Row2 + k * N] - 1;
    //     BMUvector[Row2 + k * N] = tmpSwap2;
    //   }
    //   
    //   KeyBot[j]  = Row2;
    //   KeyBot2[j] = Row1;
    // }
    // End permutation
    
    if(N > NumDataPerEpoch){
      esomwts = trainstepC3(esomwts, DataVector, BMUvector, BatchIndex, N, DIM, NumDataPerEpoch, Lines, Columns, Weights, CurrentRadius, toroid, i);
    }else{
      esomwts = trainstepC3(esomwts, DataVector, BMUvector, KeyBot, N, DIM, N, Lines, Columns, Weights, CurrentRadius, toroid, i);
    }
    
    //progress = (float) (i+1) / (float) NumEpochs;
    //int barWidth = 70;
    //std::cout << "[";
    //int pos = barWidth * progress;
    //for (int j = 0; j < barWidth; j++) {
    //  if (j < pos) std::cout << "=";
    //  else if (j == pos) std::cout << ">";
    //  else std::cout << " ";
    //}
    //std::cout << "] " << int(progress * 100.0) << " %\r";
    //std::cout.flush();
  }
  
  //std::cout << "\n";
  //std::cout << "Computations on GPU have finished." << "\n";
  
  NumericVector result(sizeESOM);
  std::copy(esomwts.begin(), esomwts.end(), result.begin());
  
  return(result);
}
