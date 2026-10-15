#include "color_distribution.h"

// #include "val_dists.h"
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "immintrin.h"

#include <cstring>

static inline int idx3(int r, int g, int b) {
  // permet d'obtenir les indices en 1d
  return (r << 6) | (g << 3) | b;
} // r*64 + g*8 + b

void ColorDistribution::reset() {
  // pour acceder au data comme tableau 1d
  //  float *data_ptr = &data[0][0][0];
  //  for (int i = 0; i < 8 * 8 * 8; i++) {
  //    data_ptr[i] = 0;
  //  }
  //  counts.fill(0);
  //  nb = 0;

  std::memset(data, 0, sizeof(data));
  counts.fill(0);
  nb = 0;
}

void ColorDistribution::add(cv::Vec3b color) {
  const int b = color[0] / 32;
  const int g = color[1] / 32;
  const int r = color[2] / 32;
  counts[idx3(r, g, b)]++;
  data[r][g][b] += 1.0f;
  nb++;
}

void ColorDistribution::finished() {
  // indique que l'échantillon est fini/complet
  // divise chaque valeur du tableau par le nombre d'échantillons
  if (nb == 0) {
    return;
  }
  const float inv = 1.0f / static_cast<float>(nb);

  // SIMDisation de la multiplication
  float *hist = &data[0][0][0];
  __m128 vinv = _mm_set1_ps(inv);

  for (int i = 0; i < 512; i += 4) {
    __m128 v = _mm_loadu_ps(hist + i);
    v = _mm_mul_ps(v, vinv);
    _mm_storeu_ps(hist + i, v);
  }
}

float ColorDistribution::distance(const ColorDistribution &other) const {
  // utiliser un tableau compilé
  //
  // NE MARCHE PAS
  //
  // float total = 0.0f;
  // for (int i = 0; i < 512; i++) {
  //   total += TAB_DIST[counts[i]][other.counts[i]];
  // }
  // return total;
  //
  //
  // Distance normale:
  // float total = 0.0f;
  // const float eps = 1e-6f;
  // const float *self_hist = &data[0][0][0];
  // const float *other_hist = &other.data[0][0][0];
  // for (int i = 0; i < 512; i++) {
  //   const float num = self_hist[i] - other_hist[i];
  //   const float den = self_hist[i] + other_hist[i] + eps;
  //   total += (num * num) / den;
  // }
  // return total;

  /*
   * Version vectorisé
   */
  const float eps = 1e-6f;

  const float *a = &data[0][0][0];
  const float *b = &other.data[0][0][0];

  __m128 acc = _mm_setzero_ps();
  const __m128 veps = _mm_set1_ps(eps);

  for (int i = 0; i < 512; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);

    __m128 num = _mm_sub_ps(va, vb);    // a - b
    __m128 num2 = _mm_mul_ps(num, num); // (a-b)^2

    __m128 den = _mm_add_ps(_mm_add_ps(va, vb), veps); // a + b + eps

    __m128 term = _mm_div_ps(num2, den); // (a-b)^2 / (a+b+eps)

    acc = _mm_add_ps(acc, term);
  }

  // somme des 4 lignes
  // s2 s1
  // s3 s0
  __m128 shuf = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 sum2 = _mm_add_ps(acc, shuf);
  shuf = _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(1, 0, 3, 2));
  __m128 sum4 = _mm_add_ps(sum2, shuf);

  return _mm_cvtss_f32(sum4);
}

ColorDistribution ColorDistribution::getColorDistribution(cv::Mat input,
                                                          cv::Point pt1,
                                                          cv::Point pt2) {
  ColorDistribution cd;
  for (int y = pt1.y; y < pt2.y; y++) {
    for (int x = pt1.x; x < pt2.x; x++) {
      cd.add(input.at<cv::Vec3b>(y, x));
    }
  }
  cd.finished();
  return cd;
}
