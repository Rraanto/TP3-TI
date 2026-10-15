#ifndef RECONNAISSANCE_H
#define RECONNAISSANCE_H

#include "histogram/histogram.h"

void calcul_histogrammes_fond(
    std::vector<histogram::ColorDistribution> &histograms,
    const cv::Mat &image);

void ajout_histogramme_objet(
    std::vector<histogram::ColorDistribution> &histograms,
    const cv::Mat &image, const cv::Point &top_left,
    const cv::Point &bottom_right, float threshold = 0.08f,
    histogram::DistanceOptions options = {});

cv::Mat renderLabels(const cv::Mat &labels,
                     const std::vector<cv::Vec3b> &colors);

#endif
