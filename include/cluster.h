#include <vector>

using namespace std;

class Cluster
{
    private:
        Image *image;
        Pixel *centroid;
                
    public:
        Cluster(Image *_image);
        ~Cluster();

        double getDistanceTo(Pixel *pixel);
        double getDistanceTo(int red, int green, int blue);
        void addPixel(Pixel *pixel);        
        double updateCentroid(int clusterId);        
        void updatePixelsList(int clusterId);
        void clearPixels();
        Pixel* getCentroid();
};
