//
// Copyright (c) 2022 xsDNN Inc. All rights reserved.
//

#ifndef XSDNN_XSDNN_H
#define XSDNN_XSDNN_H

# include <Eigen/Core>
# include <unsupported/Eigen/CXX11/Tensor>
# include "Core/Core.h"

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
# include "Utils/MnistIO.h"
# include "Utils/Except.h"

# include "Distribution/Normal.h"
# include "Distribution/Uniform.h"
# include "Distribution/Exponential.h"
# include "Distribution/Constant.h"

#endif //XSDNN_XSDNN_H
