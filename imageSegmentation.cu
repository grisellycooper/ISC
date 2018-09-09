#ifndef _IMAGESEG_KERNEL
#define _IMAGESEG_KERNEL

#include <helper_math.h>
#include <helper_functions.h>
#include "include/timer.h"

//**************** CUDA things *****************//
//Useful to read Error from CUDA Calls
#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
	printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
	printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
	exit(EXIT_FAILURE);}}
  
  
__global__ void resetClustersValues(int *d_sumRed,int *d_sumGreen,int *d_sumBlue, int* d_pixelClusterCounter, int* d_tempRedCentroid, int* d_tempGreenCentroid, int* d_tempBlueCentroid, int d_nCentroids ) {
  
	  int threadID = threadIdx.x + threadIdx.y * blockDim.x;
  
	  if(threadID < d_nCentroids) {
  
		  // nCentroids long
		  d_sumRed[threadID] = 0;
		  d_sumGreen[threadID] = 0;
		  d_sumBlue[threadID] = 0;
		  d_pixelClusterCounter[threadID] = 0;
		  d_tempRedCentroid[threadID] = 0;
		  d_tempGreenCentroid[threadID] = 0;
		  d_tempBlueCentroid[threadID] = 0;
	  }
}
  
__global__ void resetLabelArray(int *d_labelArray, int d_size){
  
	  // Global thread index
	  int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
  
	  // labelArray is "size" long
	  if(threadID < d_size) {
		  d_labelArray[threadID] = 0;
	  }
}
  
__global__ void setPixelsLabel(int *d_red, int *d_green, int *d_blue, int *d_labelArray, int d_size, int d_nCentroids, int* d_redCentroid, int* d_greenCentroid, int* d_blueCentroid ) {
  
	  // Global thread index
	  int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
  
	  double distance_pixel, distance_ccCluster;
	  int ccCluster = 0;
  
	  distance_ccCluster = sqrtf(powf((d_red[threadID]-d_redCentroid[ccCluster]),2.0) + powf((d_green[threadID]-d_greenCentroid[ccCluster]),2.0) + powf((d_blue[threadID]-d_blueCentroid[ccCluster]),2.0));
  
	  if(threadID < d_size) {
		  for(int i = 0; i < d_nCentroids; ++i) {			
			  distance_pixel = sqrtf(powf((d_red[threadID]-d_redCentroid[i]),2.0) + powf((d_green[threadID]-d_greenCentroid[i]),2.0) + powf((d_blue[threadID]-d_blueCentroid[i]),2.0));
  
			  if(distance_pixel < distance_ccCluster){
				  distance_ccCluster = distance_pixel;
				  ccCluster = i;
			  }
		  }
		  d_labelArray[threadID] = ccCluster;
	  }
}
  
__global__ void sumCluster(int *d_red, int *d_green, int *d_blue, int *d_sumRed,int *d_sumGreen, int *d_sumBlue, int *d_labelArray,int *d_pixelClusterCounter, int d_size) {
  
	  // Global thread index
	  int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
  
  
	  if(threadID < d_size) {
		  int currentLabelArray = d_labelArray[threadID];
		  int currentRed = d_red[threadID];
		  int currentGreen = d_green[threadID];
		  int currentBlue = d_blue[threadID];
		  // Writing to global memory needs a serialization.
		  atomicAdd(&d_sumRed[currentLabelArray], currentRed);
		  atomicAdd(&d_sumGreen[currentLabelArray], currentGreen);
		  atomicAdd(&d_sumBlue[currentLabelArray], currentBlue);
		  atomicAdd(&d_pixelClusterCounter[currentLabelArray], 1);
	  }
}
  
__global__ void newCentroids(int *d_tempRedCentroid, int *d_tempGreenCentroid, int *d_tempBlueCentroid,int* d_sumRed, int *d_sumGreen,int *d_sumBlue, int* d_pixelClusterCounter, int d_nCentroids) {
  
	  int threadID = threadIdx.x + threadIdx.y * blockDim.x;
  
	  if(threadID < d_nCentroids) {
		  int currentPixelCounter = d_pixelClusterCounter[threadID];
		  int sumRed = d_sumRed[threadID];
		  int sumGreen = d_sumGreen[threadID];
		  int sumBlue = d_sumBlue[threadID];
  
		  //new RGB Centroids' values written in global memory
		  d_tempRedCentroid[threadID] = (int)(sumRed/currentPixelCounter);
		  d_tempGreenCentroid[threadID] = (int)(sumGreen/currentPixelCounter);
		  d_tempBlueCentroid[threadID] = (int)(sumBlue/currentPixelCounter);
	  }
  
}

extern "C"
void executeKernel(double threshold, 
	int* h_redCentroid, int* h_greenCentroid, int* h_blueCentroid, int* d_redCentroid, int* d_greenCentroid, int* d_blueCentroid,
	int* d_sumRed, int* d_sumGreen, int* d_sumBlue, int* d_pixelClusterCounter, int*  d_tempRedCentroid, int* d_tempGreenCentroid, int* d_tempBlueCentroid,
	int* d_red, int* d_green, int* d_blue, int* h_labelArray, int* d_labelArray, size_t sizePixels, size_t sizeClusters, int d_size, int d_nCentroids)
{
	// Defining grid/block size
	double centroidChange;    
    int threadsPerBlock_ = 16;
    int gridSize = 256;
	int block_x, block_y;
	block_x = ceil((d_size + threadsPerBlock_-1)/threadsPerBlock_);
	block_y = ceil((d_size + threadsPerBlock_-1)/threadsPerBlock_);
	if (block_x > gridSize)
        block_x = gridSize;
	else if(block_y > gridSize)
		block_y = gridSize;

 	dim3 dimGrid(block_x,block_y);
    dim3 dimBlock(threadsPerBlock_,threadsPerBlock_);

    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock_);
    
	GpuTimer timer;
	timer.Start();
    	
    do
    {
        //Se ccentroids as constant
        CUDA_CALL(cudaMemcpyToSymbol(d_redCentroid, h_redCentroid, sizeClusters));
		CUDA_CALL(cudaMemcpyToSymbol(d_greenCentroid, h_greenCentroid, sizeClusters));
		CUDA_CALL(cudaMemcpyToSymbol(d_blueCentroid, h_blueCentroid, sizeClusters));
		
        //Reset values for new clusters
        resetClustersValues<<<1, dimBlock>>>(d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter, d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid);

        //Reset labelArray
        resetLabelArray<<<dimGrid, dimBlock>>>(d_labelArray, d_size);

        //Casify pixels and save value in labelArray
		setPixelsLabel<<<dimGrid, dimBlock >>> (d_red, d_green, d_blue, d_labelArray, d_size, d_nCentroids, d_redCentroid, d_greenCentroid, d_blueCentroid);

        //
        sumCluster<<<dimGrid, dimBlock>>> (d_red, d_green, d_blue, d_sumRed, d_sumGreen, d_sumBlue, d_labelArray, d_pixelClusterCounter, d_size);

        //Finds new RGB Centroids' values
		newCentroids<<<1,dimBlock>>>(d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid, d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter, d_nCentroids);

        CUDA_CALL(cudaMemcpy(h_redCentroid, d_tempRedCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_greenCentroid, d_tempGreenCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_blueCentroid, d_tempBlueCentroid, sizeClusters, cudaMemcpyDeviceToHost));
		
        centroidChange = sqrtf(powf((d_redCentroid-h_redCentroid),2.0) + powf((d_greenCentroid-h_greenCentroid),2.0) + powf((d_blueCentroid-h_blueCentroid),2.0));
		
    } while (centroidChange > threshold); 

	CUDA_CALL(cudaMemcpy(h_labelArray, d_labelArray, sizePixels, cudaMemcpyDeviceToHost));

	timer.Stop();
}

#endif