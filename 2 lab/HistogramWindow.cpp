#include "HistogramWindow.h"

#include <QtCharts>

using namespace QtCharts;

HistogramWindow::HistogramWindow(std::vector<int> &lValues, QWidget *parent)
        : QMainWindow(parent)
{
    auto vals = getRangeValues(lValues);

    auto *centralFedDistr = new QBarSet("Pixel L-value");

    for(auto &val : vals)
        *centralFedDistr << val;

    centralFedDistr->setColor(QColor::fromHslF(0.0, 0.7, 0.5));

    QBarSeries *series = new QBarSeries();
    series->append(centralFedDistr);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);

    QStringList categories;
    categories << "0-10" << "10-20" << "20-30" << "30-40"
                << "40-50" << "50-60" << "60-70" << "70-80" << "80-90" << "90-100";
    QBarCategoryAxis *axisX = new QBarCategoryAxis();
    axisX->append(categories);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis();
    axisY->setRange(0, (double)lValues.size());
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignRight);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    setCentralWidget(chartView);
    resize(800, 400);
}

std::vector<int> HistogramWindow::getRangeValues(std::vector<int> &lValues)
{
    std::vector<int> values(10, 0);

    for(auto &val : lValues)
    {
        if (val < 10)
            values[0] += 1;
        else if (val < 20)
            values[1] += 1;
        else if (val < 30)
            values[2] += 1;
        else if (val < 40)
            values[3] += 1;
        else if (val < 50)
            values[4] += 1;
        else if (val < 60)
            values[5] += 1;
        else if (val < 70)
            values[6] += 1;
        else if (val < 80)
            values[7] += 1;
        else if (val < 90)
            values[8] += 1;
        else if (val < 100)
            values[9] += 1;
    }

    return values;
}