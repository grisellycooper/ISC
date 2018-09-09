#include <string>
#include <vector>

using namespace std;

class Pixel
{
    private:
        int red;
        int green;
        int blue;
        int tag;                    /// Stores cluster ID

    public:
        Pixel(int _red, int _green, int _blue, int _tag);
        Pixel(Pixel *pixel);

        int getRed();
        int getGreen();
        int getBlue();
        void setTag(int _tag);
        int getTag();
        string getRGB();
        void setRGB(int _red, int _green, int _blue);
};

class Image
{
    private:
        int width;
        int height;
        int depth;
        vector<Pixel*> pixelsList;

    public:
        Image(string name);              /// To read
        Image(int _width, int _height);    /// To write    
        ~Image();

        void saveImage(string savePath);
        int getImageSize();
        int getImageWidth();
        int getImageHeight();        
        vector<Pixel*> getPixelsList();
        Pixel* getPixel(int index);
        Pixel* getRandomPixel();        
};