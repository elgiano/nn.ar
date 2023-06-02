// NNUGen.hpp

#pragma once
#include "NNModel.hpp"
#include "SC_PlugIn.hpp"
#include "rt_circular_buffer.h"


namespace NN {

enum NNInputs { modelIdx, methodIdx, bufferSize, inputs };

class NN : public SCUnit {

public:

  NN();
  ~NN();

  void next(int nSamples);

  NNModel* m_model;
  std::string m_method;
  int m_inDim, m_outDim;

  // BUFFERS
  int m_bufferSize;
  RTCircularBuffer<float, float>** m_inBuffer;
  RTCircularBuffer<float, float>** m_outBuffer;
  float** m_inModel;
  float** m_outModel;
private:

  void clearOutputs(int nSamples);
  bool loadModel();
  bool allocBuffers();
  void freeBuffers();

  bool m_enabled;
  bool m_useThread;

  std::unique_ptr<std::thread> m_compute_thread;
};

} // namespace NN
