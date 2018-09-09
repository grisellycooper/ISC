#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <vector>
/*#include <omp.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"*/
#include "include/image.h"
#include "include/cluster.h"
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

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

void setRGBPixels(Image* image, int* red, int* green, int* blue, int* labelArray){
    
    for (int i = 0; i < image->getImageSize(); ++i){
        //image->getPixel(i)->setRGB(h_redCentroid[h_labelArray[i]], h_greenCentroid[h_labelArray[i]], h_blueCentroid[h_labelArray[i]]);        
    }    
}


int main(int argc, char* argv[] )
{   
    ///Threads for parallel working
    //int threadCount = 4;

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

    //CUDA things
    // init device
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();
    
    // Print the vector length to be used, and compute its size
    int numPixels = image->getImageSize();

    size_t sizePixels = numPixels * sizeof(int);
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

    /*CUDA_CALL(cudaMalloc((void**) &d_red, sizePixels));
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
	CUDA_CALL(cudaMemcpyToSymbol(d_nCentroids,&nCentroids, sizeof(int)));
	CUDA_CALL(cudaMemcpyToSymbol(d_size, &size, sizeof(int)));
    */
	
    // Defining grid/block size
    int threadsPerBlock_ = 16;
    int gridSize = 256;
	int block_x, block_y;
	block_x = ceil((image->getImageWidth()+threadsPerBlock_-1)/threadsPerBlock_);
	block_y = ceil((image->getImageHeight()+threadsPerBlock_-1)/threadsPerBlock_);
	if (block_x > gridSize)
        block_x = gridSize;
	else if(block_y > gridSize)
		block_y = gridSize;

 	//dim3 dimGrid(block_x,block_y);
    //dim3 dimBlock(threadsPerBlock_,threadsPerBlock_);

    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
	//Starting timer
	//GpuTimer timer;
	//timer.Start();

    double centroidChange;
    
    do
    {
        //Se ccentroids as constant
        //CUDA_CALL(cudaMemcpyToSymbol(d_redCentroid, h_redCentroid, sizeClusters));
		//CUDA_CALL(cudaMemcpyToSymbol(d_greenCentroid, h_greenCentroid, sizeClusters));
		//CUDA_CALL(cudaMemcpyToSymbol(d_blueCentroid, h_blueCentroid, sizeClusters));
		
        //Reset values for new clusters
        //resetClustersValues<<<1, dimBlock>>>(d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter, d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid);

        //Reset labelArray
        //resetLabelArray<<<dimGrid, dimBlock>>>(d_labelArray);

        //Casify pixels and save value in labelArray
		//setPixelsLabel<<<dimGrid, dimBlock >>> (d_Red, d_Green, d_Blue, d_labelArray);

        //
        //sumCluster<<<dimGrid, dimBlock>>> (d_red, d_green, d_blue, d_sumRed, d_sumGreen, d_sumBlue, d_labelArray, d_pixelClusterCounter);

        //Finds new RGB Centroids' values
		//newCentroids<<<1,dimBlock>>>(d_tempRedCentroid, d_tempGreenCentroid, d_tempBlueCentroid, d_sumRed, d_sumGreen, d_sumBlue, d_pixelClusterCounter);

        //CUDA_CALL(cudaMemcpy(h_redCentroid, d_tempRedCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpy(h_greenCentroid, d_tempGreenCentroid, sizeClusters,cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpy(h_blueCentroid, d_tempBlueCentroid, sizeClusters, cudaMemcpyDeviceToHost));
		
        //centroidChange = sqrtf(powf((d_redCentroid-h_redCentroid),2.0) + powf((d_greenCentroid-h_greenCentroid),2.0) + powf((d_blueCentroid-h_blueCentroid),2.0));
		
    } while (centroidChange > threshold);    
    //timer.Stop();

    //CUDA_CALL(cudaMemcpy(h_labelArray, d_labelArray, sizePixels, cudaMemcpyDeviceToHost));
		
    setRGBPixels(image, h_redCentroid, h_greenCentroid, h_blueCentroid, h_labelArray);


    //******************************//
    
    /// K-means algorithm while centroidChange is less than threshold
    /*double centroidChange;
    start = clock();
    do
    {
        // Go through each pixel in the image
        int i, closestClusterIndex, imageSize = image->getImageSize(); 
        double distance_pixel, distance_ccCluster;

#       pragma omp parallel for num_threads(4) default(none) private(i,closestClusterIndex, distance_pixel, distance_ccCluster) shared(image,clusters,imageSize,clusterCount) schedule(dynamic,1)
        for (i = 0; i < imageSize ; i++)
        {            
            // Compute distance from each pixel to each cluster centroid
            int j;
            closestClusterIndex = 0;             
            distance_ccCluster = clusters[closestClusterIndex]->getDistanceTo(image->getPixel(i));
            for (j = 0; j < clusterCount; j++)
            {               
                distance_pixel = clusters[j]->getDistanceTo(image->getPixel(i));
                if (distance_pixel < distance_ccCluster)
                {
                    closestClusterIndex = j;
                    distance_ccCluster = clusters[closestClusterIndex]->getDistanceTo(image->getPixel(i));
                }
            }

            // Add tag of the nearest cluster to current pixel             
            image->getPixel(i)->setTag(closestClusterIndex);
        }        
        // Compute an average change of the centroids        
        centroidChange = 0;
        for (i = 0; i < clusterCount; i++)
        {            
            centroidChange += clusters[i]->updateCentroid(i);
        }

        centroidChange /= clusterCount;
        //cout << " - Centroid change: " << centroidChange << "\n";
    } while (centroidChange > threshold);    
    end = clock();
    cout<<"Segmentation image: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;

    // Update RGB pixels with the cluster centroid RGB
    start = clock();
    for (int i = 0; i < clusterCount; i++)
    {
        clusters[i]->updatePixelsList(i);
    }
    end = clock();
    cout<<"Final pixels updating: "<<(end - start)/(double)CLOCKS_PER_SEC <<" seconds."<< endl;
    globalTime += (end - start)/(double)CLOCKS_PER_SEC;
    */

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

	/*CUDA_CALL(cudaFree(d_Red));
	CUDA_CALL(cudaFree(d_Green));
	CUDA_CALL(cudaFree(d_Blue));
	CUDA_CALL(cudaFree(d_tempRedCentroid));
	CUDA_CALL(cudaFree(d_tempGreenCentroid));
	CUDA_CALL(cudaFree(d_tempBlueCentroid));
	CUDA_CALL(cudaFree(d_labelArray));
	CUDA_CALL(cudaFree(d_sumRed));
	CUDA_CALL(cudaFree(d_sumGreen));
	CUDA_CALL(cudaFree(d_sumBlue));
	CUDA_CALL(cudaFree(d_pixelClusterCounter));
    */
   
    cout<<"TOTAL TIME TAKEN: "<<globalTime <<" seconds."<< endl;

    //Display image with OpenCV    
    /*Mat segmentedImage = imread(saveImageAs, cv::IMREAD_COLOR);
    
    if(!segmentedImage.data ) {
        std::cout <<"Something went wrong with result image!" << std::endl ;
        return -1;
    }
  
    namedWindow( "Result image", 0);
    resizeWindow("Result image", 1920, 1080);
    imshow( "Result image", segmentedImage );        
    
    waitKey(0);*/
    return 0;
}