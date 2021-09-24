#include <QLabel>
#include "HsvSliders.h"

HsvSliders::HsvSliders(ImageViewer *imageViewer, QWidget *parent) : QWidget(parent)
{
    this->imageViewer = imageViewer;
    this->value = 0;
    this->hue = 0;
    this->saturation = 0;

    auto hueLabel = new QLabel("h", this);
    hueLabel->setGeometry(40, 0, 20, 20);
    this->hueSlider = new QSlider(this);
    this->hueSlider->setGeometry(30, 20, 30, 90);
    this->hueSlider->setMinimum(-100);
    this->hueSlider->setMaximum(100);
    this->hueSlider->setValue(0);
    connect(hueSlider, &QSlider::valueChanged, this, &HsvSliders::setHue);
    connect(this, &HsvSliders::hueChanged, this, &HsvSliders::changeImageHue);

    auto saturationLabel = new QLabel("s", this);
    saturationLabel->setGeometry(70, 0, 20, 20);
    this->saturationSlider = new QSlider(this);
    this->saturationSlider->setGeometry(60, 20, 30, 90);
    this->saturationSlider->setMinimum(-100);
    this->saturationSlider->setMaximum(100);
    this->saturationSlider->setValue(0);
    connect(saturationSlider, &QSlider::valueChanged, this, &HsvSliders::setSaturation);
    connect(this, &HsvSliders::saturationChanged, this, &HsvSliders::changeImageSaturation);

    auto valueLabel = new QLabel("v", this);
    valueLabel->setGeometry(100, 0, 20, 20);
    this->valueSlider = new QSlider(this);
    this->valueSlider->setGeometry(90, 20, 30, 90);
    this->valueSlider->setMinimum(-100);
    this->valueSlider->setMaximum(100);
    this->valueSlider->setValue(0);
    connect(valueSlider, &QSlider::valueChanged, this, &HsvSliders::setValue);
    connect(this, &HsvSliders::valueChanged, this, &HsvSliders::changeImageValue);
}

void HsvSliders::setHue(int hueVal)
{
    if (this->hue == hueVal)
        return;
    this->hue = hueVal;
    emit hueChanged(hueVal);
}

void HsvSliders::setSaturation(int saturationVal)
{
    if (this->saturation == saturationVal)
        return;
    this->saturation = saturationVal;
    emit saturationChanged(saturationVal);
}

void HsvSliders::setValue(int valueVal)
{
    if (this->value == valueVal)
        return;
    this->value = valueVal;
    emit valueChanged(valueVal);
}

void HsvSliders::changeImageSaturation(int saturationVal)
{
    for (auto x = 0; x < renderedImage.width(); x++)
        for (auto y = 0; y < renderedImage.height(); y++)
        {
            QColor color = originalImage.pixelColor(x, y);
            int up_saturation = color.hsvSaturation() * saturationVal / 100 + color.hsvSaturation();

            if (up_saturation > 255)
                up_saturation = 255;
            if (up_saturation < 0)
                up_saturation = 0;

            color.setHsv(color.hsvHue(),
                         up_saturation,
                         color.value());
            renderedImage.setPixelColor(x, y, color);
        }

    imageViewer->setPixmap(QPixmap::fromImage(renderedImage));
}

void HsvSliders::changeImageHue(int hueVal)
{
    for (auto x = 0; x < renderedImage.width(); x++)
        for (auto y = 0; y < renderedImage.height(); y++)
        {
            QColor color = originalImage.pixelColor(x, y);
            int up_hue = color.hsvHue() * hueVal / 100 + color.hsvHue();

            if (up_hue > 359)
                up_hue = 359;

            if (up_hue < 0)
                up_hue = 0;

            color.setHsv(up_hue,
                         color.hsvSaturation(),
                         color.value());
            renderedImage.setPixelColor(x, y, color);
        }

    imageViewer->setPixmap(QPixmap::fromImage(renderedImage));
}

void HsvSliders::changeImageValue(int valueVal)
{
    for (auto x = 0; x < renderedImage.width(); x++)
        for (auto y = 0; y < renderedImage.height(); y++)
        {
            QColor color = originalImage.pixelColor(x, y);
            int up_val = color.value() * valueVal / 100 + color.value();

            if (up_val > 255)
                up_val = 255;
            if (up_val < -1)
                up_val = -1;

            color.setHsv(color.hsvHue(),
                         color.hsvSaturation(),
                         up_val);
            renderedImage.setPixelColor(x, y, color);
        }

    imageViewer->setPixmap(QPixmap::fromImage(renderedImage));
}

void HsvSliders::setImage(const QImage &imageVal)
{
    HsvSliders::renderedImage = imageVal;
}

void HsvSliders::setOriginalImage(const QImage &originalImageVal)
{
    HsvSliders::originalImage = originalImageVal;
}

void HsvSliders::resetSliders()
{
    hueSlider->setValue(0);
    saturationSlider->setValue(0);
    valueSlider->setValue(0);
}
