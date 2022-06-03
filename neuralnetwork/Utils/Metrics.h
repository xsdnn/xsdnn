//##############################################################################
//#                         CALCULATE - METRICS                                #
//##############################################################################

# include "../Config.h"
# include <Eigen/Core>
# include <cmath>

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

void MSE_calculate(Matrix& target_data, Matrix& net_output, bool sqrt = false)
{
  const int target_cols = target_data.cols();
  Scalar MSE = (target_data - net_output).squaredNorm() / target_cols;
  if (sqrt) { std::cout << "RMSE Error = " << std::sqrt(MSE) << std::endl; return; }
  std::cout << "MSE Error = " << MSE << std::endl;
  return;
}

void MAE_calculate(Matrix& target_data, Matrix& net_output)
{
  Scalar MAE = (target_data - net_output).array().cwiseAbs().sum() / target_data.cols();
  std::cout << "MAE Error = " << MAE << std::endl;
}

void R_calculate(Matrix& target_data, Matrix& net_output)
{
  Scalar MSE = (target_data - net_output).squaredNorm();
  Matrix target_data_mean = target_data.rowwise().mean();

  for (int i = 0; i < target_data_mean.size(); i++)
  {
    target_data.row(i).array() -= static_cast<Scalar>(target_data_mean(i, 0));
  }
  Scalar r_coef = target_data.squaredNorm();
  std::cout << "R^2 = " << (1 - (MSE / r_coef)) << std::endl;
  return;
}
