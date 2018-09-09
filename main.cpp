// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions

#include "include/image.h"
#include "include/cluster.h"

// Constant Device Vars
__constant__ int d_nCentroids;
__constant__ int d_size;
__constant__ int d_redCentroid[2];
__constant__ int d_greenCentroid[2];
__constant__ int d_blueCentroid[2];

////////////////////////////////////////////////////////////////////////////////
// CUDA function to launch kernel
////////////////////////////////////////////////////////////////////////////////
extern "C" void  executeKernel(double threshold, 
	int* h_redCentroid, int* h_greenCentroid, int* h_blueCentroid, int* d_redCentroid, int* d_greenCentroid, int* d_blueCentroid,
	int* d_sumRed, int* d_sumGreen, int* d_sumBlue, int* d_pixelClusterCounter, int*  d_tempRedCentroid, int* d_tempGreenCentroid, int* d_tempBlueCentroid,
	int* d_red, int* d_green, int* d_blue, int* h_labelArray, int* d_labelArray, size_t sizePixels, size_t sizeClusters, int sizeImage,
    int d_nCentroids);


////////////////////////////////////////////////////////////////////////////////
// CPU functions
////////////////////////////////////////////////////////////////////////////////
void loadRGBPixels(Image* image, int* red, int* green, int* blue){
    vector<Pixel*> pixels = image->getPixelsList();

    for (int i = 0; i < image->getImageSize(); ++i){
        red[i] = pixels[i]->getRed();
        green[i] = pixels[i]->getGreen();
        blue[i] = pixels[i]->getBlue();
    }
}

void loadRGBCentroids(vector<Cluster*> clusters, int* red, int* green, int* blue){
    for(int i = 0; i < clusters.size(); ++i){
        red[i] = clusters[i]->getCentroid()->getRed();
        green[i] = clusters[i]->getCentroid()->getGreen();
        blue[i] = clusters[i]->getCentroid()->getBlue();
    }
}

void setRGBPixels(Image* image, int* h_redCentroid, int* h_greenCentroid, int* h_blueCentroid, int* labelArray){
    
    for (int i = 0; i < image->getImageSize(); ++i){
        image->getPixel(i)->setRGB(h_redCentroid[h_labelArray[i]], h_greenCentroid[h_labelArray[i]], h_blueCentroid[h_labelArray[i]]);        
    }    
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[] )
{   

    /// Time counting
	clock_t start, end;
    double globalTime = 0.0;
    
    /// Read & Write image  
    string inputImagePath, outputImagePath;

    /// Segmentation
    int clusterCount;
    double threshold = 1;             /// Umbral

    inputImagePath = argv[1];           /// Input path
    clusterCount = atoi(argv[2]);       /// Clusters

    /// Read image
    start = clock();
    Image *image(new Image(inputImagePath));
    end = clock();
    cout<<"Reading file: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;

    /// Set the output file path
    string saveImageAs = "../out/" +to_string(clusterCount) +"_" + inputImagePath.substr(9,(inputImagePath.size()));

    vector<Cluster*> clusters;
    for (int i=0; i < clusterCount; i++)
    {
        clusters.push_back(new Cluster(image));        
    }
    
    //********* CUDA things **********//
  
    // init device
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();
    
    // Print the vector length to be used, and compute its size
    int sizeImage = image->getImageSize();

    size_t sizePixels = sizeImage * sizeof(int);
    size_t sizeClusters = clusterCount * sizeof(int);

    // Allocate memory           
    //Host: Pixels' RGB values. Centroids' RGB values. Labels' array. Pixels' counter and sum
    int *h_red = (int *)malloc(sizePixels);
	int *h_green = (int *)malloc(sizePixels); 
    int *h_blue = (int *)malloc(sizePixels); 
    int *h_redCentroid = (int *)malloc(sizeClusters); 
    int *h_greenCentroid = (int *)malloc(sizeClusters); 
    int *h_blueCentroid = (int *)malloc(sizeClusters); 

    int *h_labelArray = (int *)malloc(sizePixels); 
    int *h_pixelClusterCounter = (int *)malloc(sizeClusters); 
    int *h_sumRed = (int *)malloc(sizeClusters); 
    int *h_sumGreen = (int *)malloc(sizeClusters); 
    int *h_sumBlue = (int *)malloc(sizeClusters);

    loadRGBPixels(image, h_red, h_green, h_blue);
    loadRGBCentroids(clusters, h_redCentroid, h_greenCentroid, h_blueCentroid);

   	//Device: Pixels' RGB values, Centroids' RGB values. Labels' array. Pixels' counter and sum
	int *d_red, *d_green, *d_blue, *d_tempRedCentroid, *d_tempGreenCentroid, *d_tempBlueCentroid;
    int *d_labelArray, *d_pixelClusterCounter, *d_sumRed, *d_sumGreen, *d_sumBlue;

    CUDA_CALL(cudaMalloc((void**) &d_red, sizePixels));
	CUDA_CALL(cudaMalloc((void**) &d_green, sizePixels));
	CUDA_CALL(cudaMalloc((void**) &d_blue, sizePixels));
	CUDA_CALL(cudaMalloc((void**) &d_tempRedCentroid, sizeClusters));
	CUDA_CALL(cudaMalloc((void**) &d_tempGreenCentroid, sizeClusters));
	CUDA_CALL(cudaMalloc((void**) &d_tempBlueCentroid, sizeClusters));
	CUDA_CALL(cudaMalloc((void**) &d_labelArray, sizePixels));
	CUDA_CALL(cudaMalloc((void**) &d_pixelClusterCounter, sizeClusters));
    CUDA_CALL(cudaMalloc((void**) &d_sumRed, sizeClusters));
	CUDA_CALL(cudaMalloc((void**) &d_sumGreen, sizeClusters));
	CUDA_CALL(cudaMalloc((void**) &d_sumBlue, sizeClusters));
	
	// copy host CPU memory to GPU
	CUDA_CALL(cudaMemcpy(d_red, h_red, sizePixels, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_green, h_green, sizePixels, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_blue, h_blue, sizePixels, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_tempRedCentroid, h_redCentroid,sizeClusters,cudaMemcpyHostToDevice ));
	CUDA_CALL(cudaMemcpy(d_tempGreenCentroid, h_greenCentroid,sizeClusters,cudaMemcpyHostToDevice ));
	CUDA_CALL(cudaMemcpy(d_tempBlueCentroid, h_blueCentroid,sizeClusters,cudaMemcpyHostToDevice ));
	CUDA_CALL(cudaMemcpy(d_labelArray, h_labelArray, sizePixels, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_sumRed, h_sumRed, sizeClusters, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_sumGreen, h_sumGreen, sizeClusters, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_sumBlue, h_sumBlue, sizeClusters, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_pixelClusterCounter, h_pixelClusterCounter, sizeClusters, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpyToSymbol(d_redCentroid, h_redCentroid, sizeClusters));
	CUDA_CALL(cudaMemcpyToSymbol(d_greenCentroid, h_greenCentroid, sizeClusters));
	CUDA_CALL(cudaMemcpyToSymbol(d_blueCentroid, h_blueCentroid, sizeClusters));
	CUDA_CALL(cudaMemcpyToSymbol(d_nCentroids,&clusterCount, sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(d_size, &sizeImage, sizeof(int)));
    

    executeKernel(threshold, 
                  h_redCentroid, h_greenCentroid, h_blueCentroid, d_redCentroid, d_greenCentroid, d_blueCentroid,
	              d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter, d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid,
	              d_red, d_green, d_blue, d_labelArray, h_labelArray, sizePixels, sizeClusters, d_size, d_nCentroids);
        		
    setRGBPixels(image, h_redCentroid, h_greenCentroid, h_blueCentroid, h_labelArray);

    // Save the new image
    start = clock();
    image->saveImage(saveImageAs);
    end = clock();
    cout<<"Saving result image: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;

    free(h_red);
	free(h_green);
	free(h_blue);
	free(h_redCentroid);
	free(h_greenCentroid);
	free(h_blueCentroid);
	free(h_labelArray);
	free(h_sumRed);
	free(h_sumGreen);
	free(h_sumBlue);
	free(h_pixelClusterCounter);

	CUDA_CALL(cudaFree(d_red));
	CUDA_CALL(cudaFree(d_green));
	CUDA_CALL(cudaFree(d_blue));
	CUDA_CALL(cudaFree(d_tempRedCentroid));
	CUDA_CALL(cudaFree(d_tempGreenCentroid));
	CUDA_CALL(cudaFree(d_tempBlueCentroid));
	CUDA_CALL(cudaFree(d_labelArray));
	CUDA_CALL(cudaFree(d_sumRed));
	CUDA_CALL(cudaFree(d_sumGreen));
	CUDA_CALL(cudaFree(d_sumBlue));
	CUDA_CALL(cudaFree(d_pixelClusterCounter));
    
    cout<<"TOTAL TIME TAKEN: "<<globalTime <<" seconds."<< endl;
    
    return 0;
}