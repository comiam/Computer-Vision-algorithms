#ifndef HISTOGRAMWINDOW_H
#define HISTOGRAMWINDOW_H

#include <QMainWindow>
#include <vector>

class HistogramWindow : public QMainWindow
{
Q_OBJECT

public:
    HistogramWindow(std::vector<int> &lValues, QWidget *parent = nullptr);
    ~HistogramWindow() override = default;

    std::vector<int> getRangeValues(std::vector<int> &lValues);
};
#endif
