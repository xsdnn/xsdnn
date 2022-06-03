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
