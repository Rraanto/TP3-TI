#ifndef HISTOGRAM_MODEL_H
#define HISTOGRAM_MODEL_H

#include <opencv2/core.hpp>
#include <array>
#include <cstddef>
#include <vector>

/*
 * This library implements a histogram based model for image segmentation.
 *
 * Conceptually, to predict the label of a new block, its histogram is compared
 * against a set of sets of histograms (each class has been assigned some
 * reference histograms) so that the predicted class is the nearest with respect
 * to some distance metric
 *
 * DISTANCES and BACKENDS
 * a distance metric, as defined by DistanceMetric defines (formulas) metrics
 * that can be considered for inference, for example, if two histograms are
 * represented by x1, ..., xn; y1, ..., yn, the L1 distance is simply the sum
 * of absolute differences
 *
 * a backend is somewhat a numerical implementation of a distance metric:
 * the overall default `Scalar` is one that implements the a formula "as is".
 * Alternative backends were tested to optimise the code for inference,
 * for example on x86_64 architecture, the chi-square distance can be
 * computed faster (on paper) using intrinsics (as implemented in
 * `histogram/distance_sse.cpp`)
 *
 * The core function from this (and possibly other future models used in the
 * repo) should be `::predict(const cv::Mat &, OtherParams...)` returning the
 * labeled image:
 *
 * cv::Mat predict(const cv::Mat &bgr, const ReferenceBank &references,
                PredictionOptions options = {});
 */

namespace histogram {

// Types of distances that can be used for inference
enum class DistanceMetric { ChiSquare, L1 };
enum class Backend { Scalar, SSE };

// model parameters
struct DistanceOptions {
  DistanceMetric metric = DistanceMetric::ChiSquare;
  Backend backend = Backend::Scalar;
};

struct PredictionOptions {
  int block_size = 8;
  DistanceOptions distance;
};

class ColorDistribution {
public:
  static constexpr std::size_t bin_count = 512;

  // Build a normalized histogram from a nonempty ROI in a BGR CV_8UC3 image.
  static ColorDistribution fromImage(const cv::Mat &bgr, cv::Rect roi);
  const std::array<float, bin_count> &bins() const noexcept;

private:
  explicit ColorDistribution(std::array<float, bin_count> normalized_bins);
  std::array<float, bin_count> bins_;
};

// Outer indices are class IDs; class 0 conventionally represents background.
using ReferenceBank = std::vector<std::vector<ColorDistribution>>;

bool isAvailable(DistanceOptions options) noexcept;
float distance(const ColorDistribution &a, const ColorDistribution &b,
               DistanceOptions options = {});

// Insert when the nearest reference is at least admission_threshold away.
bool addReference(std::vector<ColorDistribution> &references,
                  const ColorDistribution &candidate,
                  float admission_threshold = 0.08f,
                  DistanceOptions options = {});

// Row-major tiles, including partial border tiles.
std::vector<ColorDistribution> backgroundReferences(const cv::Mat &bgr,
                                                    int tile_size = 128);

// Return image-sized CV_32SC1 labels without modifying inputs. Exact ties use
// the lowest class ID; partial border blocks are classified at their real size.
cv::Mat predict(const cv::Mat &bgr, const ReferenceBank &references,
                PredictionOptions options = {});

} // namespace histogram
#endif
