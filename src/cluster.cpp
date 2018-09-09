#include <iostream>
#include <cmath>
#include "../include/image.h"
#include "../include/cluster.h"

using namespace std;

Cluster::Cluster(Image *_image)
{
    image = _image;
    centroid = new Pixel(image->getRandomPixel()); 
} 

Cluster::~Cluster()
{
    delete centroid;
}

double Cluster::getDistanceTo(Pixel *pixel)
{
    int dRed = centroid->getRed() - pixel->getRed();
    int dGreen = centroid->getGreen() - pixel->getGreen();
    int dBlue = centroid->getBlue() - pixel->getBlue();
    double distance = sqrt(pow(dRed, 2) + pow(dGreen, 2) + pow(dBlue, 2));  ///Euclidean distance
    
    return distance;
}

double Cluster::getDistanceTo(int red, int green, int blue)
{
    int dRed = centroid->getRed() - red;
    int dGreen = centroid->getGreen() - green;
    int dBlue = centroid->getBlue() - blue;
    double distance = sqrt(pow(dRed, 2) + pow(dGreen, 2) + pow(dBlue, 2));  ///Euclidean distance

    return distance;
}

void Cluster::addPixel(Pixel *pixel)
{
    //pixelsList.push_back(pixel);
}

double Cluster::updateCentroid(int clusterId)
{
    double aRed = 0;
    double aGreen = 0;
    double aBlue = 0;
    int imageSize = image->getImageSize();  
    double change = 0;
    int i, count = 0;

#   pragma omp parallel for num_threads(4) default(none) reduction(+:count) private(i) shared(imageSize, clusterId, image, centroid) schedule(dynamic,10) reduction(+:count)
    for (i = 0; i < imageSize; i++)
    {
        if(image->getPixel(i)->getTag() == clusterId)
        {
            aRed += image->getPixel(i)->getRed();
            aGreen += image->getPixel(i)->getGreen();
            aBlue += image->getPixel(i)->getBlue();
            count++;
        }        
    }

    if (count < 1)
    {
        count = 1;
    }

    aRed /= count;
    aGreen /= count;
    aBlue /= count;
    change = this->getDistanceTo(aRed, aGreen, aBlue);
    centroid->setRGB(aRed, aGreen, aBlue);
    return change;
}

void Cluster::updatePixelsList(int clusterId)
{
    int i, imageSize = image->getImageSize();
//#   pragma omp parallel for num_threads(4) default(none) private(i) shared(imageSize, clusterId, image, centroid) schedule(dynamic,10)
    for (i = 0; i < imageSize; i++)
    {
        if(image->getPixel(i)->getTag() == clusterId)
        {
            image->getPixel(i)->setRGB(centroid->getRed(), centroid->getGreen(), centroid->getBlue());
        }        
    }
}

Pixel* Cluster::getCentroid()
{
    return centroid;
}

void Cluster::clearPixels()
{
   // pixelsList = {};
}