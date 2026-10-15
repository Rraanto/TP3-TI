/*
 * implementation of the distance backend with x86_64 vectorisation
 */

#include "histogram.h"
#include <xmmintrin.h>

namespace histogram::detail {
float chiSquareSSE(const float *a, const float *b) {
  __m128 acc = _mm_setzero_ps();
  const __m128 eps = _mm_set1_ps(1e-6f);
  for (std::size_t i = 0; i < ColorDistribution::bin_count; i += 4) {
    const __m128 va = _mm_loadu_ps(a + i);
    const __m128 vb = _mm_loadu_ps(b + i);

    // xi - yi
    const __m128 difference = _mm_sub_ps(va, vb);

    // xi + yi + eps
    const __m128 denominator = _mm_add_ps(_mm_add_ps(va, vb), eps);
    acc = _mm_add_ps(
        acc, _mm_div_ps(_mm_mul_ps(difference, difference), denominator));
  }
  __m128 shuffled = _mm_shuffle_ps(acc, acc, _MM_SHUFFLE(2, 3, 0, 1));
  const __m128 pairs = _mm_add_ps(acc, shuffled);
  shuffled = _mm_shuffle_ps(pairs, pairs, _MM_SHUFFLE(1, 0, 3, 2));
  return _mm_cvtss_f32(_mm_add_ps(pairs, shuffled));
}
} // namespace histogram::detail
