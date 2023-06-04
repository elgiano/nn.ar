// NNUGens.hpp

#pragma once
#include "NNModel.hpp"
#include "SC_PlugIn.hpp"
#include "rt_circular_buffer.h"

namespace NN {

using RingBuf = RingBufCtrl<float, float>;

class NN : public SCUnit {
public:

  NN();
  ~NN();

  void next(int nSamples);


  NNModel* m_model;
  NNModelMethod* m_method;
  int m_inDim, m_outDim;

  int m_bufferSize;
  float* m_inModel;
  float* m_outModel;

private:
  enum UGenInputs { modelIdx=0, methodIdx, bufSize, inputs };
  void clearOutputs(int nSamples);
  bool loadModel();
  bool allocBuffers();
  void freeBuffers();

  RingBuf* m_inBuffer;
  RingBuf* m_outBuffer;
  bool m_enabled;
  bool m_useThread;

  std::unique_ptr<std::thread> m_compute_thread;
};


class NNParamUGen : public SCUnit {
public:
  NNParamUGen();

  enum NNParamInputs { modelIdx=0, settingIdx };
  NNModel* m_model;
  unsigned short m_setting;
};

class NNSet : public NNParamUGen {
public:
  NNSet();

  void next(int nSamples);

private:
  enum UGenInputs { value=NNParamInputs::settingIdx+1 };
};

class NNGet : public NNParamUGen {
public:
  NNGet();

  void next(int nSamples);
};

} // namespace NN
