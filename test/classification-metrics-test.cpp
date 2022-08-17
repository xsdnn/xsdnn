# include "../neuralnetwork/xsDNN.h"
typedef Eigen::MatrixXd Matrix;

/*
 * Тестируется корректный подсчет основных метрик классификации.
 *
 * Тест пройден.
 */

int main()
{
    const int sample_size = 9;
    Matrix y_true(1, sample_size), y_pred(1, sample_size);
    y_true << 0, 0, 1, 1, 0, 0, 1, 1, 0;
    y_pred << 0, 0, 1, 1, 0, 0, 1, 1, 1;
    metrics::accuracy_precision_recall_f1_calculate(y_true, y_pred);
    return 0;
}
