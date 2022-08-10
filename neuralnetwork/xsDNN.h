#pragma once

# include <Eigen/Core>

# include "NeuralNetwork.h"

# include "RNG.h"

# include "Config.h"




# include "Activation/ReLU.h"
# include "Activation/LeakyReLU.h"
# include "Activation/Sigmoid.h"
# include "Activation/Softmax.h"
# include "Activation/Identity.h"



# include "Layer.h"
# include "Layer/FullyConnected.h"
# include "Layer/Dropout.h"
# include "Layer/BatchNormalization.h"



# include "Optimizer.h"
# include "Optimizer/SGD.h"


# include "Output.h"
# include "Output/RegressionMSE.h"
# include "Output/BinaryClassEntropy.h"
# include "Output/MultiClassEntropy.h"


# include "Utils/Metrics.h"
# include "Utils/InOut.h"
# include "Utils/mnist_io.h"

# include "Distribution/Normal.h"
# include "Distribution/Uniform.h"
# include "Distribution/Exponential.h"