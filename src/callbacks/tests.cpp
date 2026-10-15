#include "tests.h"

#include <iostream>

#include "histogram/histogram.h"

using namespace cv;
using namespace std;

void calcul_distance_gauche_droite(const Mat &img_input) {
  if (img_input.cols < 2 || img_input.empty())
    return;
  const int middle = img_input.cols / 2;
  const auto left = histogram::ColorDistribution::fromImage(
      img_input, Rect(0, 0, middle, img_input.rows));
  const auto right = histogram::ColorDistribution::fromImage(
      img_input, Rect(middle, 0, img_input.cols - middle, img_input.rows));
  const float d = histogram::distance(left, right);
  cout << "Distance de couleurs: " << d << endl;
}

void test_blocs_nb_et_distance(Mat &img_input, const Mat &frozen,
                               bool &config_1) {
  /*
   * effectue le test sur l'image partiulière de blocs noir et blancs
   */
  if (frozen.empty())
    return;

  img_input = frozen.clone();

  const int width = img_input.cols;
  const int height = img_input.rows;
  const Rect roiGH(0, 0, width / 2, height / 2);                  // gauche haut
  const Rect roiGB(0, height / 2, width / 2, height / 2);         // gauche bas
  const Rect roiDH(width / 2, 0, width / 2, height / 2);          // droite haut
  const Rect roiDB(width / 2, height / 2, width / 2, height / 2); // droite bas
  const Mat noir(height / 2, width / 2, CV_8UC3, Vec3b(0, 0, 0));
  const Mat blanc(height / 2, width / 2, CV_8UC3, Vec3b(255, 255, 255));

  blanc.copyTo(img_input(roiGH));
  blanc.copyTo(img_input((config_1) ? roiDB : roiDH));

  noir.copyTo(img_input(roiGB));
  noir.copyTo(img_input((config_1) ? roiDH : roiDB));

  config_1 = !config_1;

  calcul_distance_gauche_droite(img_input);
}
