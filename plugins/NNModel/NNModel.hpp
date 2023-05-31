// PluginNNModel.hpp
// Gianluca Elia (elgiano@gmail.com)

#pragma once

#include "SC_PlugIn.hpp"

namespace NN {

class NNModel : public SCUnit {
public:
    NNModel();

    // Destructor
    // ~NNModel();

private:
    // Calc function
    void next(int nSamples);

    // Member variables
};

} // namespace NN
