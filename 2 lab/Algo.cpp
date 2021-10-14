//
// Created by maxim on 07.10.2021.
//

#include <cmath>
#include <iostream>
#include <utility>
#include "Algo.h"

//void prepareMatrix(double matrix[3][3]) {
//    int sum = 0;
//    for (int i = 0; i < 3; ++i)
//        for (int j = 0; j < 3; ++j)
//            sum += matrix[i][j];
//
//    for (int i = 0; i < 3; ++i)
//        for (int j = 0; j < 3; ++j)
//            matrix[i][j] /= sum;
//}

int sobelMatrixX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
};

int sobelMatrixY[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1},
};

int range(int min, int max, int val)
{
    if (val < min)
        val = min;
    else if (val > max)
        val = max;

    return val;
}

inline double intensivity(int r, int g, int b)
{
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

std::tuple<bool, int, int, int> getPixelByCoords(QImage &image, int xb, int yb, int moveX, int moveY)
{
    if (xb + moveX < 0 || xb + moveX >= image.width())
        return std::tuple(false, 0, 0, 0);
    if (yb + moveY < 0 || yb + moveY >= image.height())
        return std::tuple(false, 0, 0, 0);

    auto pix = image.pixelColor(xb + moveX, yb + moveY);
    return std::tuple(true, pix.red(), pix.green(), pix.blue());
}

double getPixelIByCoords(QImage &image, int xb, int yb, int moveX, int moveY)
{
    auto arg = getPixelByCoords(image, xb, yb, moveX, moveY);

    if (!std::get<0>(arg))
        return 0;

    return intensivity(std::get<1>(arg), std::get<2>(arg), std::get<3>(arg));
}

std::tuple<QImage, std::vector<std::vector<int>>> Algo::sobelFilter(QImage *img)
{
    std::vector<std::vector<int>> vectors(image.width(), std::vector<int>(image.height(), 0));
    QImage clone = img ? img->copy() : image.copy();
    for (auto x = 0; x < clone.width(); x++)
        for (auto y = 0; y < clone.height(); y++)
        {
            double Gx = 0, Gy = 0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    Gx += sobelMatrixX[i][j] * getPixelIByCoords(image, x, y, i - 1, j - 1);
                    Gy += sobelMatrixY[i][j] * getPixelIByCoords(image, x, y, i - 1, j - 1);
                }
            int newPix = int(sqrt(pow(Gx, 2) + pow(Gy, 2)));
            newPix = range(0, 255, newPix);

            int angle = (int) ((atan2(Gy, Gx) * 180 / M_PI + 180));
            int roundedAngle = angle - angle % 45;
            if (roundedAngle >= 180)
                roundedAngle -= 180;

            vectors[x][y] = roundedAngle;

            clone.setPixel(x, y, QColor(newPix, newPix, newPix).rgb());
        }
    imageViewer->setPixmap(QPixmap::fromImage(clone));

    return std::tuple(clone, vectors);
}

double gaussDist(int x, int y, double sigma)
{
    double expDegree = -(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2));

    return exp(expDegree) / (2 * M_PI * pow(sigma, 2));
}

QImage Algo::gaussFilter(int kSize, double sigma)
{
    double gaussKernel[kSize][kSize];
    int center = kSize / 2;
    double sum = 0.0;

    for (auto i = 0; i < kSize; ++i)
        for (auto j = 0; j < kSize; ++j)
        {
            gaussKernel[i][j] = gaussDist(i - center, j - center, sigma);
            sum += gaussKernel[i][j];
        }

    for (auto i = 0; i < kSize; ++i)
        for (auto j = 0; j < kSize; ++j)
            gaussKernel[i][j] /= sum;

    QImage clone = image.copy();
    for (auto x = 0; x < clone.width(); x++)
        for (auto y = 0; y < clone.height(); y++)
        {
            double r = 0;
            double g = 0;
            double b = 0;
            for (int i = 0; i < kSize; ++i)
                for (int j = 0; j < kSize; ++j)
                {
                    auto pixCoords = getPixelByCoords(image, x, y, i - center, j - center);
                    if (!std::get<0>(pixCoords))
                        continue;

                    r += std::get<1>(pixCoords) * gaussKernel[i][j];
                    g += std::get<2>(pixCoords) * gaussKernel[i][j];
                    b += std::get<3>(pixCoords) * gaussKernel[i][j];
                }
            int rValue = range(0, 255, int(r));
            int gValue = range(0, 255, int(g));
            int bValue = range(0, 255, int(b));
            clone.setPixel(x, y, QColor(rValue, gValue, bValue).rgb());
        }

    imageViewer->setPixmap(QPixmap::fromImage(clone));

    return clone;
}

double gaborKernelFunc(double xS, double yS, double gamma, double lambda, double phi, double sigma)
{
    return exp(-(pow(xS, 2) + gamma * pow(yS, 2)) / (2 * pow(sigma, 2))) * cos(2 * M_PI * xS / lambda + phi);
}

void Algo::gaborFilter(int kSize, double gamma, double lambda, double theta, double phi)
{
    double gaborKernel[kSize][kSize];
    int center = kSize / 2;
    theta *= 180 / M_PI;//to radians
    double sigma = 0.56 * lambda;

    double sum = 0.0;

    for (auto i = 0; i < kSize; ++i)
    {
        int xPos = i - center;
        for (auto j = 0; j < kSize; ++j)
        {
            int yPos = j - center;
            double xS = xPos * cos(theta) + yPos * sin(theta);
            double yS = -xPos * sin(theta) + yPos * cos(theta);
            gaborKernel[i][j] = gaborKernelFunc(xS, yS, gamma, lambda, phi, sigma);
            sum += gaborKernel[i][j];
        }
    }
    for (auto i = 0; i < kSize; ++i)
        for (auto j = 0; j < kSize; ++j)
            gaborKernel[i][j] /= sum;

    QImage clone = image.copy();
    for (auto x = 0; x < clone.width(); x++)
        for (auto y = 0; y < clone.height(); y++)
        {
            sum = 0;
            for (int i = 0; i < kSize; ++i)
                for (int j = 0; j < kSize; ++j)
                    sum += gaborKernel[i][j] * getPixelIByCoords(image, x, y, i - center, j - center);

            int iValue = range(0, 255, int(sum));
            clone.setPixel(x, y, QColor(iValue, iValue, iValue).rgb());
        }

    imageViewer->setPixmap(QPixmap::fromImage(clone));
}

void Algo::setImage(QImage img)
{
    this->image = std::move(img);
}

void Algo::setImageViewer(ImageViewer *_imageViewer)
{
    this->imageViewer = _imageViewer;
}

QImage nonMaxSuppression(QImage image, std::vector<std::vector<int>> vectors)
{
    QImage clone = image.copy(1, 1, image.width() - 2, image.height() - 2);
    QImage tmp = clone.copy();

    for (auto x = 1; x < clone.width() - 1; ++x)
        for (auto y = 1; y < clone.height() - 1; ++y)
        {
            int bottomPixel;
            int topPixel;
            int basePixel = tmp.pixelColor(x, y).red();
            int angle = vectors[x][y];
            if (!angle)
            {
                bottomPixel = tmp.pixelColor(x, y - 1).red();
                topPixel = tmp.pixelColor(x, y + 1).red();
            } else if (angle == 45)
            {
                bottomPixel = tmp.pixelColor(x + 1, y - 1).value();
                topPixel = tmp.pixelColor(x - 1, y + 1).value();
            } else if (angle == 90)
            {
                bottomPixel = tmp.pixelColor(x - 1, y).value();
                topPixel = tmp.pixelColor(x + 1, y).value();
            } else
            {
                bottomPixel = tmp.pixelColor(x - 1, y - 1).value();
                topPixel = tmp.pixelColor(x + 1, y + 1).value();
            }

            if (basePixel > bottomPixel && basePixel > topPixel)
                clone.setPixel(x, y, tmp.pixelColor(x, y).rgb());
            else
                clone.setPixel(x, y, QColor(0, 0, 0).rgb());
        }

    return clone;
}

QImage doubleThresholding(QImage image, int lowThreshold, int highThreshold)
{
    auto clone = image.copy();

    for (auto i = 0; i < clone.width(); ++i)
        for (auto j = 0; j < clone.height(); ++j)
        {
            int value = image.pixelColor(i, j).red();
            if (value >= highThreshold)
                value = 255;
            else if (value < lowThreshold)
                value = 0;

            clone.setPixel(i, j, QColor(value, value, value).rgb());
        }

    return clone;
}

int getGuaranteedPixelVal(QImage image, int x, int y)
{
    return image.pixelColor(x, y).red();
}

QImage hysteresis(QImage image)
{
    auto clone = image.copy();

    for (auto x = 1; x < clone.width() - 1; ++x)
        for (auto y = 1; y < clone.height() - 1; ++y)
            if (
                    (getGuaranteedPixelVal(image, x + 1, y - 1) == 255) ||
                    (getGuaranteedPixelVal(image, x + 1, y) == 255) ||
                    (getGuaranteedPixelVal(image, x + 1, y + 1) == 255) ||
                    (getGuaranteedPixelVal(image, x, y - 1) == 255) ||
                    (getGuaranteedPixelVal(image, x, y + 1) == 255) ||
                    (getGuaranteedPixelVal(image, x - 1, y - 1) == 255) ||
                    (getGuaranteedPixelVal(image, x - 1, y) == 255) ||
                    (getGuaranteedPixelVal(image, x - 1, y + 1) == 255)
                    )
                clone.setPixel(x, y, QColor(255, 255, 255).rgb());
            else
                clone.setPixel(x, y, QColor(0, 0, 0).rgb());

    return clone;
}

void Algo::canny(int kSize, double sigma, int lowThreshold, int highThreshold)
{
    auto img = gaussFilter(kSize, sigma);
    auto res = sobelFilter(&img);
    img = nonMaxSuppression(std::get<0>(res), std::get<1>(res));
    img = doubleThresholding(img, lowThreshold, highThreshold);
    img = hysteresis(img);

    imageViewer->setPixmap(QPixmap::fromImage(img));
}

int otsuThreshold(std::vector<int> &grayVector)
{
    auto[min, max] = std::minmax_element(begin(grayVector), end(grayVector));
    if (*min - *max == 0)
        return -1;

    int histSize = *max - *min + 1;
    int *hist = new int[histSize]{0};

    for (int &i: grayVector)
        hist[i - *min]++;

    double m = 0; // m - сумма высот всех бинов, домноженных на положение их середины
    double n = 0; // n - сумма высот всех бинов
    for (int t = 0; t <= *max - *min; t++)
    {
        m += t * hist[t];
        n += hist[t];
    }

    double maxSigma = -1; // Максимальное значение межклассовой дисперсии
    int threshold = 0; // Порог, соответствующий maxSigma

    int alpha1 = 0; // Сумма высот всех бинов для класса 1
    int beta1 = 0; // Сумма высот всех бинов для класса 1, домноженных на положение их середины

    // Переменная alpha2 не нужна, т.к. она равна m - alpha1
    // Переменная beta2 не нужна, т.к. она равна n - alpha1
    for (int t = 0; t < *max - *min; t++)
    {
        alpha1 += t * hist[t];
        beta1 += hist[t];

        // Считаем вероятность класса 1.
        double w1 = beta1 / n;
        // Нетрудно догадаться, что w2 тоже не нужна, т.к. она равна 1 - w1

        // a = a1 - a2, где a1, a2 - средние арифметические для классов 1 и 2
        double a = ((double) alpha1) / beta1 - ((double) (m - alpha1)) / (n - beta1);

        double sigma = w1 * (1 - w1) * a * a;
        if (sigma > maxSigma)
        {
            maxSigma = sigma;
            threshold = t;
        }
    }

    return threshold + *min;
}

void Algo::otsu()
{
    std::vector<int> grayVector;
    QImage clone = image.copy();

    for (auto i = 0; i < image.width(); ++i)
        for (auto j = 0; j < image.height(); ++j)
        {
            auto pixel = image.pixelColor(i, j);
            int iVal = (int)intensivity(pixel.red(), pixel.green(), pixel.blue());
            clone.setPixel(i, j, QColor(iVal, iVal, iVal).rgb());
            grayVector.emplace_back(iVal);
        }

    auto threshold = otsuThreshold(grayVector);
    for (auto x = 0; x < image.width(); ++x)
        for (auto y = 0; y < image.height(); ++y)
        {
            auto pix = clone.pixelColor(x, y).red();
            pix = pix < threshold ? 0 : 255;

            clone.setPixel(x, y, QColor(pix, pix, pix).rgb());
        }

    imageViewer->setPixmap(QPixmap::fromImage(clone));
}