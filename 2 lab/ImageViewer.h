#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QLabel>
#include <QMouseEvent>

class ImageViewer : public QLabel
{
Q_OBJECT
private:
    QLabel *schemeColors;
public:
    ImageViewer(QWidget *parent = nullptr);
    void mouseMoveEvent(QMouseEvent *me);
    ~ImageViewer() override = default;
};

#endif
