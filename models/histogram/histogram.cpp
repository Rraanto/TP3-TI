#include "histogram.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace histogram {
#ifdef HISTOGRAM_HAS_SSE
namespace detail {
float chiSquareSSE(const float *a, const float *b);
}
#endif

namespace {
using Kernel = float (*)(const float *, const float *);

float chiSquare(const float *a, const float *b) {
  float total = 0.0f;
  for (std::size_t i = 0; i < ColorDistribution::bin_count; ++i) {
    const float difference = a[i] - b[i];
    total += difference * difference / (a[i] + b[i] + 1e-6f);
  }
  return total;
}

float l1(const float *a, const float *b) {
  float total = 0.0f;
  for (std::size_t i = 0; i < ColorDistribution::bin_count; ++i)
    total += std::abs(a[i] - b[i]);
  return total;
}

Kernel selectKernel(DistanceOptions options) {
  if (!isAvailable(options))
    throw std::runtime_error(
        "Requested histogram metric/backend is unavailable");
#ifdef HISTOGRAM_HAS_SSE
  if (options.backend == Backend::SSE)
    return detail::chiSquareSSE;
#endif
  return options.metric == DistanceMetric::L1 ? l1 : chiSquare;
}

void validateImage(const cv::Mat &image) {
  if (image.empty() || image.dims != 2 || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected a nonempty 2D BGR CV_8UC3 image");
}
} // namespace

ColorDistribution::ColorDistribution(std::array<float, bin_count> bins)
    : bins_(std::move(bins)) {}

ColorDistribution ColorDistribution::fromImage(const cv::Mat &bgr,
                                               cv::Rect roi) {
  validateImage(bgr);
  if (roi.x < 0 || roi.y < 0 || roi.width <= 0 || roi.height <= 0 ||
      roi.x > bgr.cols || roi.y > bgr.rows || roi.width > bgr.cols - roi.x ||
      roi.height > bgr.rows - roi.y)
    throw std::invalid_argument(
        "Histogram ROI must be nonempty and inside the image");

  std::array<std::uint64_t, bin_count> counts{};
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    const auto *row = bgr.ptr<cv::Vec3b>(y);
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      const auto &pixel = row[x];
      const int index =
          ((pixel[2] / 32) << 6) | ((pixel[1] / 32) << 3) | (pixel[0] / 32);
      ++counts[index];
    }
  }
  const double pixels = static_cast<double>(roi.width) * roi.height;
  std::array<float, bin_count> bins{};
  for (std::size_t i = 0; i < bin_count; ++i)
    bins[i] = static_cast<float>(counts[i] / pixels);
  return ColorDistribution(bins);
}

const std::array<float, ColorDistribution::bin_count> &
ColorDistribution::bins() const noexcept {
  return bins_;
}

bool isAvailable(DistanceOptions options) noexcept {
  if (options.metric != DistanceMetric::ChiSquare &&
      options.metric != DistanceMetric::L1)
    return false;
  if (options.backend == Backend::Scalar)
    return true;
#ifdef HISTOGRAM_HAS_SSE
  return options.backend == Backend::SSE &&
         options.metric == DistanceMetric::ChiSquare;
#else
  return false;
#endif
}

float distance(const ColorDistribution &a, const ColorDistribution &b,
               DistanceOptions options) {
  return selectKernel(options)(a.bins().data(), b.bins().data());
}

bool addReference(std::vector<ColorDistribution> &references,
                  const ColorDistribution &candidate, float threshold,
                  DistanceOptions options) {
  if (!std::isfinite(threshold) || threshold < 0)
    throw std::invalid_argument(
        "Admission threshold must be finite and nonnegative");
  const auto kernel = selectKernel(options);
  for (const auto &reference : references)
    if (kernel(reference.bins().data(), candidate.bins().data()) < threshold)
      return false;
  references.push_back(candidate);
  return true;
}

std::vector<ColorDistribution> backgroundReferences(const cv::Mat &bgr,
                                                    int tile_size) {
  validateImage(bgr);
  if (tile_size <= 0)
    throw std::invalid_argument("Tile size must be positive");
  std::vector<ColorDistribution> references;
  for (int y = 0; y < bgr.rows;) {
    const int height = std::min(tile_size, bgr.rows - y);
    for (int x = 0; x < bgr.cols;) {
      const int width = std::min(tile_size, bgr.cols - x);
      references.push_back(
          ColorDistribution::fromImage(bgr, {x, y, width, height}));
      x += width;
    }
    y += height;
  }
  return references;
}

cv::Mat predict(const cv::Mat &bgr, const ReferenceBank &references,
                PredictionOptions options) {
  validateImage(bgr);
  if (options.block_size <= 0)
    throw std::invalid_argument("Block size must be positive");
  if (references.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      std::none_of(references.begin(), references.end(),
                   [](const auto &group) { return !group.empty(); }))
    throw std::invalid_argument(
        "Expected reference histograms with representable class IDs");
  const auto kernel = selectKernel(options.distance);
  cv::Mat labels(bgr.size(), CV_32SC1);
  for (int y = 0; y < bgr.rows;) {
    const int height = std::min(options.block_size, bgr.rows - y);
    for (int x = 0; x < bgr.cols;) {
      const int width = std::min(options.block_size, bgr.cols - x);
      const cv::Rect roi(x, y, width, height);
      const auto query = ColorDistribution::fromImage(bgr, roi);
      float best = std::numeric_limits<float>::infinity();
      int label = 0;
      for (std::size_t id = 0; id < references.size(); ++id) {
        for (const auto &reference : references[id]) {
          const float value =
              kernel(query.bins().data(), reference.bins().data());
          if (value < best) {
            best = value;
            label = static_cast<int>(id);
          }
        }
      }
      labels(roi).setTo(label);
      x += width;
    }
    y += height;
  }
  return labels;
}
} // namespace histogram
