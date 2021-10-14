#include "ImageViewer.h"
#include "colorsUtils.h"

ImageViewer::ImageViewer(QWidget *parent) : QLabel(parent) {
    this->schemeColors = new QLabel(parent);
    this->setScaledContents(true);
    this->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

    schemeColors->setText("rgb: r: g: b:\n"
                          "hsv: h: s: v: \n"
                          "lab: l: a: b:");

    QRect a = QRect(800, 20, 150, 70);
    schemeColors->setGeometry(a);
    setMouseTracking(true);
}

void ImageViewer::mouseMoveEvent(QMouseEvent *me) {
    auto pos = me->pos();

    if(!this->pixmap())
        return;

    auto img_pixmap = this->pixmap()->toImage();
    auto img_pixel = img_pixmap.pixel(pos.x(), pos.y());
    auto color = QColor(img_pixel);
    int r;
    int g;
    int b;
    color.getRgb(&r, &g, &b);
    RgbColor rgb;
    rgb.r = r;
    rgb.g = g;
    rgb.b = b;
    HsvColor hsvColor = RgbToHsv(rgb);
    LabColor labColor;
    XyzColor xyzColor;
    xyzColor = rgbToXyz(rgb);
    labColor = xyzToLab(xyzColor);
    QString printable = QString("r: %1, g: %2,  b: %3\n"
                                "h: %4, s: %5, v: %6\n"
                                "l: %7, a: %8 b: %9").arg(QString::number(rgb.r), QString::number(rgb.g),
                                                          QString::number(rgb.g), QString::number(hsvColor.h),
                                                          QString::number(hsvColor.s), QString::number(hsvColor.v),
                                                          QString::number(labColor.l, 'g', 2),
                                                          QString::number(labColor.a, 'g', 2),
                                                          QString::number(labColor.b, 'g', 2));
    this->schemeColors->setText(printable);
}
