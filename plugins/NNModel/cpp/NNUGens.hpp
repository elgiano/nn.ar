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
  bool m_should_stop_perform_thread;
  std::binary_semaphore m_data_available_lock, m_result_available_lock;


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


class NNAttrUGen : public SCUnit {
public:
  NNAttrUGen();

  enum NNAttrInputs { modelIdx=0, attrIdx };
  NNModel* m_model;
  unsigned short m_attrIdx;
};

class NNSet : public NNAttrUGen {
public:
  NNSet();
  void next(int nSamples);

private:
  enum UGenInputs { value=NNAttrInputs::attrIdx+1 };
};

class NNGet : public NNAttrUGen {
public:
  NNGet();
  void next(int nSamples);
};

} // namespace NN
