
//#include "trainstepc3.h"

#include "CL/cl.hpp"

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

std::vector<float> trainstepC3(std::vector<float> esomwts,
                               std::vector<float> DataSampled,
                               std::vector<float> BMUsampled,
                               std::vector<int> Index,
                               int N, int DIM, int NumDataPerEpoch,
                               int Lines, int Columns, int Weights, int Radius,
                               bool toroid, int Iteration){
  
  // OpenCL device search ------------------------------------------------------
  vector<cl::Device> cl_AllDevices; // get all devices of all platforms
  vector<cl::Platform> cl_AllPlatforms; // get all platforms (drivers)
  cl::Platform::get(&cl_AllPlatforms);
  
  for(u_int i= 0 ;i <(u_int)cl_AllPlatforms.size(); i++) {
    vector<cl::Device> cl_devices_available;
    cl_AllPlatforms[i].getDevices(CL_DEVICE_TYPE_ALL, &cl_devices_available);
    for(u_int j=0; j<(u_int)cl_devices_available.size(); j++){
      cl_AllDevices.push_back(cl_devices_available[j]);
    }
  }
  
  cl::Device cl_device; // select fastest available device
  
  if(cl_AllDevices.size() == 0){
    //std::cout << "No OpenCL capable device detected." << "\n";
    return std::vector<float>();
  }
  
  int MaxNumCUs     = cl_AllDevices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  u_int BestDeviceIdx = 0;
  
  if(cl_AllDevices.size() > 1){
    for(u_int i = 0; i < (u_int)cl_AllDevices.size(); i++){
      cl_device = cl_AllDevices[i];
      int tmpNumCUs = cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
      
      if(tmpNumCUs > MaxNumCUs){
        MaxNumCUs     = tmpNumCUs;
        BestDeviceIdx = i;
      }
    }
  }
  
  cl_device = cl_AllDevices[BestDeviceIdx];
  
  //if(Iteration == 0){
    //std::cout << "Chosen device: " << cl_device.getInfo<CL_DEVICE_NAME>() << "\n";
    //std::cout << "Device Version: " << cl_device.getInfo<CL_DRIVER_VERSION>() << "\n";
  //}
  
  // OpenCL device search END --------------------------------------------------
  
  
  int LCS = Lines * Columns;
  float Factor = 1.0;
  
  std::vector<float> DataRow(DIM);
  
  if((N >= 2501) && (Radius <= 16)){
    if(Radius <= 16 && Radius > 8){
      Factor = 0.75;
    }else if (Radius <= 8 && Radius > 4){
      Factor = 0.5;
    }else{
      Factor = 0.1;
    }
  }
  
  std::string KernelUpdateWeights = get_UpdateWeights();
  
  cl::Context MyOCLContext;
  cl::CommandQueue MyOCLQueue;
  
  // Build and compile the kernel
  MyOCLContext = cl::Context(cl_device);
  MyOCLQueue   = cl::CommandQueue(MyOCLContext, cl_device, CL_QUEUE_PROFILING_ENABLE);
  
  //if(toroid == true){
    std::string KernelToroidDistance = get_ToroidDistance();
    cl::Program Program_ToroidDistance;
    cl::Program::Sources cl_source_ToroidDistance;
    cl_source_ToroidDistance.push_back({KernelToroidDistance.c_str(), KernelToroidDistance.length()});
    Program_ToroidDistance = cl::Program(MyOCLContext, cl_source_ToroidDistance);
    int errorA = Program_ToroidDistance.build("-cl-fast-relaxed-math -w");
    cl::Kernel kernel_ToroidDistance(Program_ToroidDistance, "toroid_distance");
    cl::NDRange range_TD(Lines, Columns);
  //}else{
    std::string KernelNonToroidDistance = get_NonToroidDistance();
    cl::Program Program_NonToroidDistance;
    cl::Program::Sources cl_source_NonToroidDistance;
    cl_source_NonToroidDistance.push_back({KernelNonToroidDistance.c_str(), KernelNonToroidDistance.length()});
    Program_NonToroidDistance = cl::Program(MyOCLContext, cl_source_NonToroidDistance);
    int errorB = Program_NonToroidDistance.build("-cl-fast-relaxed-math -w");
    cl::Kernel kernel_NonToroidDistance(Program_NonToroidDistance, "non_toroid_distance");
    cl::NDRange range_NTD(Lines, Columns);
  //}
  
  cl::Program Program_UpdateWeights;
  cl::Program::Sources cl_source_UpdateWeights;
  cl_source_UpdateWeights.push_back({KernelUpdateWeights.c_str(), KernelUpdateWeights.length()});
  Program_UpdateWeights = cl::Program(MyOCLContext, cl_source_UpdateWeights);
  int error2 = Program_UpdateWeights.build("-cl-fast-relaxed-math -w");
  cl::Kernel kernel_UpdateWeights(Program_UpdateWeights, "update_weights");
  cl::NDRange range_UW(Lines, Columns, Weights);
  
  // Initiate/Instantiate OpenCL buffers:
  // Create OpenCL buffers for static/const variables
  cl::Buffer clData(MyOCLContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * DIM, DataSampled.data());
  
  // Create OpenCL buffers for changing variables
  cl::Buffer clESOM(MyOCLContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * Lines*Columns*Weights, esomwts.data());
  cl::Buffer clDM(MyOCLContext, CL_MEM_READ_WRITE, sizeof(float) * Lines*Columns);
  
  //for(int p = 0; p < N; p++){
  for(int p = 0; p < NumDataPerEpoch; p++){
    
    int DataIdx = Index[p];
    
    float bmpos0 = BMUsampled[DataIdx];
    float bmpos1 = BMUsampled[DataIdx + N];
    
    if(toroid == true){
      // Set kernel arguments
      kernel_ToroidDistance.setArg(0, bmpos0);
      kernel_ToroidDistance.setArg(1, bmpos1);
      kernel_ToroidDistance.setArg(2, (float) Lines - 1);
      kernel_ToroidDistance.setArg(3, (float) Columns - 1);
      kernel_ToroidDistance.setArg(4, Lines);
      kernel_ToroidDistance.setArg(5, Columns);
      kernel_ToroidDistance.setArg(6, LCS);
      kernel_ToroidDistance.setArg(7, clDM);
      MyOCLQueue.enqueueNDRangeKernel(kernel_ToroidDistance, cl::NullRange, range_TD, cl::NullRange);
    }else{
      // Set kernel arguments
      kernel_NonToroidDistance.setArg(0, bmpos0);
      kernel_NonToroidDistance.setArg(1, bmpos1);
      kernel_NonToroidDistance.setArg(2, Lines);
      kernel_NonToroidDistance.setArg(3, Columns);
      kernel_NonToroidDistance.setArg(4, clDM);
      MyOCLQueue.enqueueNDRangeKernel(kernel_NonToroidDistance, cl::NullRange, range_NTD, cl::NullRange);
    }
    
    
    // Set kernel arguments
    kernel_UpdateWeights.setArg(0, clESOM);
    kernel_UpdateWeights.setArg(1, clData);
    kernel_UpdateWeights.setArg(2, clDM);
    kernel_UpdateWeights.setArg(3, DataIdx); // RowIdx of data row => Data[p, ]
    kernel_UpdateWeights.setArg(4, N);
    kernel_UpdateWeights.setArg(5, Lines);
    kernel_UpdateWeights.setArg(6, Columns);
    kernel_UpdateWeights.setArg(7, Weights);
    kernel_UpdateWeights.setArg(8, Radius);
    kernel_UpdateWeights.setArg(9, Factor);
    MyOCLQueue.enqueueNDRangeKernel(kernel_UpdateWeights, cl::NullRange, range_UW, cl::NullRange);
    MyOCLQueue.finish();
  }
  
  //MyOCLQueue.finish();
  MyOCLQueue.enqueueReadBuffer(clESOM, CL_TRUE, 0, sizeof(float) * Lines*Columns*Weights, esomwts.data());
  
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
