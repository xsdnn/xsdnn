#pragma once

# include <Eigen/Core>

# include "NeuralNetwork.h"

# include "RNG.h"

# include "Config.h"




# include "Activation/ReLU.h"
# include "Activation/Sigmoid.h"
# include "Activation/Softmax.h"
# include "Activation/Identity.h"



# include "Layer.h"
# include "Layer/FullyConnected.h"



# include "Optimizer.h"
# include "Optimizer/SGD.h"


# include "Output.h"
# include "Output/RegressionMSE.h"
# include "Output/BinaryClassEntropy.h"
# include "Output/MultiClassEntropy.h"



# include "Callback.h"
# include "Callback/VerboseCallback.h"

# include "Utils/Metrics.h"
# include "Utils/DataLoader.h"
# include "Utils/InOut.h"

# include "Utils/mnist/mnist_reader.hpp"
# include "Utils/mnist/mnist_reader_common.hpp"
# include "Utils/mnist/mnist_reader_less.hpp"
# include "Utils/mnist/mnist_utils.hpp"


# include "Distribution/Normal.h"
# include "Distribution/Uniform.h"
# include "Distribution/Exponential.h"