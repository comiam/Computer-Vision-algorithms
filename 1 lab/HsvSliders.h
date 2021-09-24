#ifndef HSVSLIDER_H
#define HSVSLIDER_H

#include <QWidget>
#include <QSlider>
#include "QDebug"
#include "ImageViewer.h"

class HsvSliders : public QWidget
{
Q_OBJECT

public slots:
    void setHue(int hueVal);
    void setSaturation(int saturationVal);
    void setValue(int valueVal);
    void setImage(const QImage &imageVal);

private:
    ImageViewer *imageViewer;
    int hue{};
    int saturation{};
    int value{};
    QSlider *hueSlider;
    QSlider *saturationSlider;
    QSlider *valueSlider;
    QImage renderedImage;
    QImage originalImage;

    void changeImageSaturation(int saturationVal);
    void changeImageHue(int hueVal);
    void changeImageValue(int valueVal);
public:
    void resetSliders();
    void setOriginalImage(const QImage &originalImageVal);
    HsvSliders(ImageViewer *imageViewer, QWidget *parent);

signals:
    void hueChanged(int hue);
    void saturationChanged(int saturation);
    void valueChanged(int value);
};


#endif
