#ifndef INC_1_LAB_ALGO_H
#define INC_1_LAB_ALGO_H

#include <QImage>
#include "ImageViewer.h"

class Algo {
public:
    std::tuple<QImage, std::vector<std::vector<int>>> sobelFilter(QImage *img);
    QImage gaussFilter(int kSize, double sigma);
    void gaborFilter(int kSize, double gamma, double lambda, double theta, double phi);
    void setImage(QImage img);
    void canny(int kSize, double sigma, int lowThreshold, int highThreshold);
    void setImageViewer(ImageViewer *imageViewer);
    void otsu();
private:
    QImage image;
    ImageViewer *imageViewer;
};

#endif
