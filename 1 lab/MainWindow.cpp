#include "MainWindow.h"
#include "colorsUtils.h"
#include <QAction>
#include <QMenuBar>
#include <QApplication>
#include <QStandardPaths>
#include <QFileDialog>
#include <QMessageBox>
#include <QImageReader>
#include <fstream>
#include <QImageWriter>
#include <QString>

MainWindow::MainWindow(const QString &title, int minW, int minH, QWidget *parent) : QMainWindow(parent)
{
    this->setFixedSize(minW, minH);
    this->setWindowTitle(title);

    this->imageViewer = new ImageViewer(this);
    this->imageViewer->show();
    this->imageViewer->setGeometry(0, 0, int(minH), int(minH));

    this->hsvSlider = new HsvSliders(this->imageViewer, this);
    this->hsvSlider->setMinimumSize(200, 200);
    this->hsvSlider->setGeometry(820, 90, 300, 300);

    this->histogramButton = new QPushButton(this);
    this->histogramButton->setText("L-histogram");
    this->histogramButton->setGeometry(800, 200, 200, 50);
    connect(this->histogramButton, &QPushButton::clicked, this, &MainWindow::onPressHistogram);

    QMenu *imageMenu;
    imageMenu = menuBar()->addMenu("Image");
    auto *saveImage = new QAction("save img", this);
    auto *loadImage = new QAction("load img", this);
    connect(saveImage, &QAction::triggered, this, &MainWindow::onPressSaveImage);
    connect(loadImage, &QAction::triggered, this, &MainWindow::onPressLoadImage);
    imageMenu->addAction(loadImage);
    imageMenu->addAction(saveImage);
}

void MainWindow::onPressLoadImage()
{
    QString qStrFilePath = QFileDialog::getOpenFileName(this, tr("Open Image"), QStandardPaths::writableLocation(
            QStandardPaths::PicturesLocation), tr("Image Files (*.png *.jpg *.bmp)"));
    QImageReader reader(qStrFilePath);

    if (qStrFilePath.isEmpty())
        return;

    if (!reader.canRead())
    {
        showMessageBox(QString("can't read renderedImage"));
        return;
    }

    QImage image = reader.read();
    image = image.scaled(imageViewer->width(), imageViewer->height(),
                         Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    hsvSlider->setImage(image);
    hsvSlider->setOriginalImage(image);
    hsvSlider->resetSliders();
    imageViewer->setPixmap(QPixmap::fromImage(image));
}

void MainWindow::onPressSaveImage()
{
    QString qStrFilePath = QFileDialog::getSaveFileName(this,
                                                        tr("Save Image"),
                                                        QStandardPaths::writableLocation(
                                                                QStandardPaths::PicturesLocation)),
            tr("JPG file (*.jpg);;PNG file (*.png);;BMP file (*.bmp)");

    if (qStrFilePath.isEmpty())
        return;

    QImageWriter writer(qStrFilePath);

    if (!writer.canWrite())
        showMessageBox(QString("can't write fie"));
    else
        writer.write(imageViewer->pixmap()->toImage());
}

void MainWindow::showMessageBox(const QString &msgText)
{
    QMessageBox messageBox;
    messageBox.setText(msgText);
    messageBox.exec();
}

void MainWindow::onPressHistogram()
{
    if (!imageViewer->pixmap())
        return;

    auto image = imageViewer->pixmap()->toImage();
    std::vector<int> hist_list{};
    auto width = image.width();
    auto height = image.height();
    int r, g, b;
    RgbColor rgb;
    XyzColor xyz;
    LabColor lab;
    int value;

    for (auto x = 0; x < width; ++x)
    {
        for (auto y = 0; y < height; ++y)
        {
            auto img_pixel = image.pixel(x, y);
            auto color = QColor(img_pixel);
            color.getRgb(&r, &g, &b);
            rgb.r = r;
            rgb.g = g;
            rgb.b = b;
            xyz = rgbToXyz(rgb);
            lab = xyzToLab(xyz);
            value = int(lab.l);
            hist_list.emplace_back(value);
        }
    }

    histogramWindow = std::make_unique<HistogramWindow>(hist_list);
    histogramWindow->show();
}
