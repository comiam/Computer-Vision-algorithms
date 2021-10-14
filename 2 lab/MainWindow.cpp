#include "MainWindow.h"
#include "colorsUtils.h"
#include "Algo.h"
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
#include <QInputDialog>

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

    QMenu *filtersMenu = menuBar()->addMenu("Filters");
    auto *sobelFilter = new QAction("Apply Sobel", this);
    connect(sobelFilter, &QAction::triggered, this, &MainWindow::onSobelFilter);
    filtersMenu->addAction(sobelFilter);

    auto *gaussFilter = new QAction("Apply Gauss", this);
    connect(gaussFilter, &QAction::triggered, this, &MainWindow::onGaussFilter);
    filtersMenu->addAction(gaussFilter);

    auto *gaborFilter = new QAction("Apply Gabor", this);
    connect(gaborFilter, &QAction::triggered, this, &MainWindow::onGaborFilter);
    filtersMenu->addAction(gaborFilter);

    QMenu *algoMenu = menuBar()->addMenu("Algorithms");
    auto *canny = new QAction("Apply Canny", this);
    connect(canny, &QAction::triggered, this, &MainWindow::onCanny);
    algoMenu->addAction(canny);

    auto *otsu = new QAction("Apply Otsu Binarization", this);
    connect(otsu, &QAction::triggered, this, &MainWindow::onOtsu);
    algoMenu->addAction(otsu);

    algos = new Algo();
    algos->setImageViewer(imageViewer);
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
    algos->setImage(image);
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

void MainWindow::onGaussFilter()
{
    if (!imageViewer->pixmap())
        return;

    bool bOk;
    QString str = QInputDialog::getText(this,
                                        "Input",
                                        "Kernel size sigma:",
                                        QLineEdit::Normal,
                                        "5, 1.4",
                                        &bOk
    );
    if (!bOk)
        return;

    auto strs = str.split(',');
    std::for_each(strs.begin(), strs.end(), [](QString &key)
    {
        key = key.trimmed();
    });

    auto kSize = strs[0].toInt();
    auto sigma = strs[1].toDouble();

    algos->gaussFilter(kSize, sigma);
}

void MainWindow::onSobelFilter()
{
    if (!imageViewer->pixmap())
        return;
    algos->sobelFilter(nullptr);
}

void MainWindow::onGaborFilter()
{
    if (!imageViewer->pixmap())
        return;

    bool bOk;
    QString str = QInputDialog::getText(this,
                                        "Input",
                                        "Kernel size, gamma, lambda, theta, phi:",
                                        QLineEdit::Normal,
                                        "5, 0.1, 2, 45, 0",
                                        &bOk
    );
    if (!bOk)
        return;

    auto strs = str.split(',');
    std::for_each(strs.begin(), strs.end(), [](QString &key)
    {
        key = key.trimmed();
    });

    auto kSize = strs[0].toInt();
    auto gamma = strs[1].toDouble();
    auto lambda = strs[2].toDouble();
    auto theta = strs[3].toInt();
    auto phi = strs[4].toInt();

    algos->gaborFilter(kSize, gamma, lambda, theta, phi);
}

void MainWindow::onCanny()
{
    if (!imageViewer->pixmap())
        return;

    bool bOk;
    QString str = QInputDialog::getText(this,
                                        "Input",
                                        "Kernel size, sigma, upThr, downThr:",
                                        QLineEdit::Normal,
                                        "5, 0.1, 50, 100",
                                        &bOk
    );
    if (!bOk)
        return;

    auto strs = str.split(',');
    std::for_each(strs.begin(), strs.end(), [](QString &key)
    {
        key = key.trimmed();
    });

    auto kSize = strs[0].toInt();
    auto sigma = strs[1].toDouble();
    auto upThr = strs[2].toInt();
    auto downThr = strs[3].toInt();

    algos->canny(kSize, sigma, downThr, upThr);
}

void MainWindow::onOtsu()
{
    if (!imageViewer->pixmap())
        return;

    algos->otsu();
}
