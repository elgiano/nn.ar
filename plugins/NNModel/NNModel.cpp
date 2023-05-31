// PluginNNModel.cpp
// Gianluca Elia (elgiano@gmail.com)

#include "SC_PlugIn.hpp"
#include "NNModel.hpp"

static InterfaceTable* ft;

namespace NN {

NNModel::NNModel() {
    mCalcFunc = make_calc_function<NNModel, &NNModel::next>();
    next(1);
}

void NNModel::next(int nSamples) {
    const float* input = in(0);
    const float* gain = in(1);
    float* outbuf = out(0);

    // simple gain function
    for (int i = 0; i < nSamples; ++i) {
        outbuf[i] = input[i] * gain[i];
    }
}

} // namespace NN

PluginLoad(NNModelUGens) {
    // Plugin magic
    ft = inTable;
    registerUnit<NN::NNModel>(ft, "NNModel", false);
}
