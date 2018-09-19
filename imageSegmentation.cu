#ifndef _IMAGESEG_KERNEL
#define _IMAGESEG_KERNEL

//#include <helper_math.h>
//#include <helper_functions.h>
#include <cstdio>
#include "timer.h"

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
  	  
	if(threadID < d_size) {
		distance_ccCluster = sqrtf(powf((d_red[threadID]-d_redCentroid[ccCluster]),2.0) + powf((d_green[threadID]-d_greenCentroid[ccCluster]),2.0) + powf((d_blue[threadID]-d_blueCentroid[ccCluster]),2.0));
  		
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
	double centroidChange = 2;    
    int threadsPerBlock_ = 16;
    int gridSize = 256;
	/*int block_x, block_y;
	block_x = ceil((d_size + threadsPerBlock_-1)/threadsPerBlock_); // TOCHECK
	block_y = block_x;

	//printf("%d threads per block\n", block_x);

	if (block_x > gridSize)
        block_x = gridSize;
	else if(block_y > gridSize)
		block_y = gridSize;*/

 	dim3 dimGrid(gridSize,gridSize);
    dim3 dimBlock(threadsPerBlock_,threadsPerBlock_);

    printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock_);
    
	printf("umbral:  %lf \n", threshold);
	printf("d_size:  %d \n", d_size);
	printf("d_nCentroids:  %d \n", d_nCentroids);

	/*printf("**h_labelArray:  %d \n", h_labelArray[0]); 
	printf("**h_labelArray:  %d \n", h_labelArray[1]); */
	//printf("red:  %d \n", *d_red);
	
	/*for(int i = 0; i < 5; ++i){
        printf("red:  %d \n", d_red[i]);
        printf("green:  %d \n", d_green[i]);
        printf("blue:  %d \n", d_blue[i]);
    }
*/
    for(int i = 0; i < d_nCentroids ; ++i){
        printf("redCentroid:  %d \n", h_redCentroid[i]);
        printf("greenCentroid:  %d \n", h_greenCentroid[i]);
        printf("blueCentroid:  %d \n", h_blueCentroid[i]);                
    }   

	GpuTimer timer;
	timer.Start();

	//TEST
	int *h_red_ = (int *)malloc(sizePixels);
	int *h_green_ = (int *)malloc(sizePixels); 
    int *h_blue_ = (int *)malloc(sizePixels); 
    int *h_redCentroid_ = (int *)malloc(sizeClusters); 
    int *h_greenCentroid_ = (int *)malloc(sizeClusters); 
    int *h_blueCentroid_ = (int *)malloc(sizeClusters); 

    int *h_labelArray_ = (int *)malloc(sizePixels); 
    int *h_pixelClusterCounter_ = (int *)malloc(sizeClusters); 
    int *h_sumRed_ = (int *)malloc(sizeClusters); 
    int *h_sumGreen_ = (int *)malloc(sizeClusters); 
    int *h_sumBlue_ = (int *)malloc(sizeClusters);
    	
    /*do
    {*/
        //Se ccentroids as constant
        /*CUDA_CALL(cudaMemcpyToSymbol(d_redCentroid, h_redCentroid, sizeClusters));
		CUDA_CALL(cudaMemcpyToSymbol(d_greenCentroid, h_greenCentroid, sizeClusters));
		CUDA_CALL(cudaMemcpyToSymbol(d_blueCentroid, h_blueCentroid, sizeClusters));*/

		/*CUDA_CALL(cudaMemcpy(d_redCentroid, h_redCentroid, sizeClusters, cudaMemcpyHostToDevice));
    	CUDA_CALL(cudaMemcpy(d_greenCentroid, h_greenCentroid, sizeClusters, cudaMemcpyHostToDevice));
    	CUDA_CALL(cudaMemcpy(d_blueCentroid, h_blueCentroid, sizeClusters, cudaMemcpyHostToDevice));*/
		
        //Reset values for new clusters
        resetClustersValues<<<1, dimBlock>>>(d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter, d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid, d_nCentroids);
		
		/*CUDA_CALL(cudaMemcpy(h_sumRed_, d_sumRed, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_sumGreen_, d_sumGreen, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_sumBlue_, d_sumBlue, sizeClusters, cudaMemcpyDeviceToHost))
		CUDA_CALL(cudaMemcpy(h_pixelClusterCounter_, d_pixelClusterCounter, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_redCentroid_, d_tempRedCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_greenCentroid_, d_tempGreenCentroid, sizeClusters, cudaMemcpyDeviceToHost))
		CUDA_CALL(cudaMemcpy(h_blueCentroid_, d_tempBlueCentroid, sizeClusters,cudaMemcpyDeviceToHost));

		for(int j = 0; j<d_nCentroids; ++j){
			printf("Centroid %d \n", j); 
			printf("h_sumRed:  %d \n", h_sumRed_[j]); 
			printf("h_sumGreen:  %d \n", h_sumGreen_[j]); 
			printf("h_sumBlue:  %d \n", h_sumBlue_[j]); 
			printf("h_pixelClusterCounter:  %d \n", h_pixelClusterCounter_[j]); 
			printf("h_redCentroid:  %d \n", h_redCentroid_[j]); 
			printf("h_greenCentroid:  %d \n", h_greenCentroid_[j]); 
			printf("h_blueCentroid:  %d \n", h_blueCentroid_[j]); 
			printf("------ \n"); 
		}*/
		
        //Reset labelArray
        resetLabelArray<<<dimGrid, dimBlock>>>(d_labelArray, d_size);

		/*CUDA_CALL(cudaMemcpy(h_labelArray_, d_labelArray, sizePixels,cudaMemcpyDeviceToHost));
		printf("------- \n");
		for(int j = 0; j<10; ++j){			 
			printf("h_labelArray:  %d \n", h_labelArray_[j]); 			
		}
		printf("------ \n"); */

        //Casify pixels and save value in labelArray
		setPixelsLabel<<<dimGrid, dimBlock >>> (d_red, d_green, d_blue, d_labelArray, d_size, d_nCentroids, d_redCentroid, d_greenCentroid, d_blueCentroid);
		
		CUDA_CALL(cudaMemcpy(d_red, d_sumRed, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(d_green, d_sumGreen, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(d_blue, d_sumBlue, sizeClusters, cudaMemcpyDeviceToHost))
		CUDA_CALL(cudaMemcpy(h_pixelClusterCounter_, d_pixelClusterCounter, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_redCentroid_, d_tempRedCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_greenCentroid_, d_tempGreenCentroid, sizeClusters, cudaMemcpyDeviceToHost))
		CUDA_CALL(cudaMemcpy(h_blueCentroid_, d_tempBlueCentroid, sizeClusters,cudaMemcpyDeviceToHost));

		for(int j = 0; j<d_nCentroids; ++j){
			printf("Centroid %d \n", j); 
			printf("h_sumRed:  %d \n", h_sumRed_[j]); 
			printf("h_sumGreen:  %d \n", h_sumGreen_[j]); 
			printf("h_sumBlue:  %d \n", h_sumBlue_[j]); 
			printf("h_pixelClusterCounter:  %d \n", h_pixelClusterCounter_[j]); 
			printf("h_redCentroid:  %d \n", h_redCentroid_[j]); 
			printf("h_greenCentroid:  %d \n", h_greenCentroid_[j]); 
			printf("h_blueCentroid:  %d \n", h_blueCentroid_[j]); 
			printf("------ \n"); 
		}

        //
        //sumCluster<<<dimGrid, dimBlock>>> (d_red, d_green, d_blue, d_sumRed, d_sumGreen, d_sumBlue, d_labelArray, d_pixelClusterCounter, d_size);
		
		/*int *h_pixelClusterCounter = (int *)malloc(sizeClusters);

		h_pixelClusterCounter[0] = 0;
		h_pixelClusterCounter[1] = 0;

		printf("h_sumRed:  %d \n", h_pixelClusterCounter[0]); 
		printf("h_sumRed:  %d \n", h_pixelClusterCounter[1]);

		
		//CUDA_CALL(cudaMemcpy(h_pixelClusterCounter, d_pixelClusterCounter, sizeClusters, cudaMemcpyDeviceToHost));

		printf("h_sumRed:  %d \n", h_pixelClusterCounter[0]); 
		printf("h_sumRed:  %d \n", h_pixelClusterCounter[1]);*/

		//printf("CUDA kernel launch with %d blocks of %d threads\n", gridSize, threadsPerBlock_);
        //Finds new RGB Centroids' values
		//newCentroids<<<1,dimBlock>>>(d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid, d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter, d_nCentroids);

        /*CUDA_CALL(cudaMemcpy(h_redCentroid, d_tempRedCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_greenCentroid, d_tempGreenCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(h_blueCentroid, d_tempBlueCentroid, sizeClusters, cudaMemcpyDeviceToHost));*/
		
		/*printf("h_redCentroid:  %d \n", h_redCentroid[0]); 
		printf("h_redCentroid:  %d \n", h_redCentroid[1]); */

        // centroidChange = sqrtf(powf((d_redCentroid-h_redCentroid),2.0) + powf((d_greenCentroid-h_greenCentroid),2.0) + powf((d_blueCentroid-h_blueCentroid),2.0));
	    centroidChange--;
		printf("centroidChange:  %f \n", centroidChange);
		
    //} while (centroidChange > threshold); 
	
	//CUDA_CALL(cudaMemcpy(h_labelArray, d_labelArray, sizePixels, cudaMemcpyDeviceToHost));	

	/*printf("h_labelArray:  %d \n", h_labelArray[0]); 
	printf("h_labelArray:  %d \n", h_labelArray[1]); 
*/

	timer.Stop();
}

#endif