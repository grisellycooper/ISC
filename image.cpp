#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "image.h"

using namespace std;

Pixel::Pixel(int _red, int _green, int _blue, int _tag)
{
    red = _red;
    green = _green;
    blue = _blue;   
    tag = _tag;
}

Pixel::Pixel(Pixel *pixel)
{
    red = pixel->red;
    green = pixel->green;
    blue = pixel->blue;
}

int Pixel::getRed()
{
    return red;
}

int Pixel::getGreen()
{
    return green;
}

int Pixel::getBlue()
{
    return blue;
}

void Pixel::setTag(int _tag)
{
    tag = _tag;
}

int Pixel::getTag()
{
    return tag;
}

string Pixel::getRGB()
{
    return to_string(red) + " " + to_string(green) + " " + to_string(blue);
}

void Pixel::setRGB(int _red, int _green, int _blue)
{
    red = _red;
    green = _green;
    blue = _blue;
}

Image::Image(int _width, int _height)
{
    width = _width;
    height = _height;
    for (int i=0; i < (width*height); i++)
    {
        pixelsList.push_back(new Pixel(0, 0, 0, 0));
    }
}

Image::Image(string imageDir)
{
    ifstream image(imageDir);
    if (image)
    {
        string type;
        image >> type;
        if (type == "P3")
        {
            int red;
            int green;
            int blue;
            image >> width;
            image >> height;
            image >> depth;
            for (int i = 0; i < (width * height); i++)
            {
                image >> red;
                image >> green;
                image >> blue;
                pixelsList.push_back(new Pixel(red, green, blue, 0));
            }
        } else {
            cout << "The format file must be .ppm type P3" << endl;
        }
    } else {
        cout << "File not found!" << endl;
    }
}

Image::~Image()
{
    for (int i = 0; i < width * height; i++)
    {
        delete pixelsList[i];
    }
}

void Image::saveImage(string name)
{
    ofstream image(name);
    Pixel *pixel = NULL;
    if (image)
    {
        image << "P3" << "\n";
        image << width << " " << height << "\n";
        image << depth << "\n";
        for (int y = 0; y < width; y++)
        {
            for (int x = 0; x < height; x++)
            {
                pixel = pixelsList[height * y + x];
                image << pixel->getRGB() << " ";
            }
            image << "\n";
        }
    } else {
        cout << name << "Something went wrong with the file " <<name << "\n";
    }
}

Pixel* Image::getRandomPixel()
{
    return pixelsList[rand() % (width * height)];    
}

vector<Pixel*> Image::getPixelsList()
{
    return pixelsList;
}

Pixel* Image::getPixel(int index)
{
    return pixelsList[index];
}

int Image::getImageSize()
{
    return width * height;
}

int Image::getImageWidth()
{
    return width;
}

int Image::getImageHeight()
{
    return height;
}