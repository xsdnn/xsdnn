#pragma once

# include <Eigen/Core>

# include "NeuralNetwork.h"

# include "RNG.h"

# include "Config.h"




# include "Activation/ReLU.h"
# include "Activation/Sigmoid.h"



# include "Layer.h"
# include "Layer/FullyConnected.h"



# include "Optimizer.h"
# include "Optimizer/SGD.h"



# include "Output.h"
# include "Output/RegressionMSE.h"
# include "Output/BinaryClassEntropy.h"



# include "Callback.h"
# include "Callback/VerboseCallback.h"
# include "Callback/RegressorCallback.h"
# include "Callback/ClassificationCallback.h"