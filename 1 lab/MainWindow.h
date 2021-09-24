#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QPushButton>
#include "ImageViewer.h"
#include "HsvSliders.h"
#include "HistogramWindow.h"
#include <memory>

class MainWindow : public QMainWindow {
Q_OBJECT

private:
    ImageViewer *imageViewer;
    HsvSliders *hsvSlider;
    QPushButton *histogramButton;

    void showMessageBox(const QString &msgText);
    std::unique_ptr<HistogramWindow> histogramWindow;

public:
    MainWindow(const QString &title, int minW, int minH, QWidget *parent = nullptr);

private slots:
    void onPressHistogram();
    void onPressLoadImage();
    void onPressSaveImage();
};

#endif
