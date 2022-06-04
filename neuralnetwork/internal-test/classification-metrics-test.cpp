# include "../DNN.h"
# include <iostream>
# include <vector>
# include <cmath>
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

// TODO: помыть жопу

int main()
{
  // generate data
  const int sample_size = 9;
  Matrix y_true(1, sample_size), y_pred(1, sample_size);
  y_true << 0, 0, 1, 1, 0, 0, 1, 1, 0;
  y_pred << 0, 0, 1, 1, 0, 0, 1, 1, 1;
  accuracy_precision_recall_f1_calculate(y_true, y_pred);
  return 0;
}
