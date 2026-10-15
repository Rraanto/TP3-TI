#include "reconnaissance.h"
#include <iostream>

void calcul_histogrammes_fond(
    std::vector<histogram::ColorDistribution> &histograms,
    const cv::Mat &image) {
  histograms = histogram::backgroundReferences(image);
  std::cout << "[b] histogrammes fonds : " << histograms.size() << '\n';
}

void ajout_histogramme_objet(
    std::vector<histogram::ColorDistribution> &histograms,
    const cv::Mat &image, const cv::Point &top_left,
    const cv::Point &bottom_right, float threshold,
    histogram::DistanceOptions options) {
  const auto candidate = histogram::ColorDistribution::fromImage(
      image, cv::Rect(top_left, bottom_right));
  if (histogram::addReference(histograms, candidate, threshold, options))
    std::cout << "[a] histogrammes objets : " << histograms.size() << '\n';
  else
    std::cout << "[a] histogramme rejeté\n";
}

cv::Mat renderLabels(const cv::Mat &labels,
                     const std::vector<cv::Vec3b> &colors) {
  cv::Mat rendered(labels.size(), CV_8UC3);
  for (int y = 0; y < labels.rows; ++y) {
    const auto *ids = labels.ptr<int>(y);
    auto *pixels = rendered.ptr<cv::Vec3b>(y);
    for (int x = 0; x < labels.cols; ++x) {
      const int id = ids[x];
      pixels[x] = id >= 0 && static_cast<std::size_t>(id) < colors.size()
                      ? colors[id] : cv::Vec3b(0, 0, 0);
    }
  }
  return rendered;
}
