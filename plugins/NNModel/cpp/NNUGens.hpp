// NNUGens.hpp

#pragma once
#include "NNModel.hpp"
#include "backend/backend.h"
#include "SC_PlugIn.hpp"
#include "rt_circular_buffer.h"
#include <chrono>
#include <semaphore>
#include <string>
#include <thread>

namespace NN {

using RingBuf = RingBufCtrl<float, float>;

enum Debug { none=0, attributes=1, all=2 };

class NNSetAttr {
public:
  std::string attrName;
  // remember in0 indices
  int inputIdx;

  NNSetAttr(std::string name, int inputIdx, float initVal);

  // called in audio thread: check trig, update value and flag
  void update(Unit* unit, int nSamples);

  bool changed() const { return valUpdated; }
  // called before model_perform
  std::string getStrValue() {
    valUpdated = false;
    return std::to_string(value);
  }

private:
  float lastTrig = 0;
  float value = 0;
  bool valUpdated = false;
};

class NN : public SCUnit {
public:

  NN();
  ~NN();

  void next(int nSamples);
  void freeBuffers();
  void setupAttributes();
  void warmupModel();

  float* m_inModel;
  float* m_outModel;
  NNModelDesc* m_modelDesc;
  NNModelMethod* m_method;
  Backend m_model;
  int m_inDim, m_outDim;
  int m_bufferSize, m_debug;
  std::vector<NNSetAttr> m_attributes;
  std::binary_semaphore m_data_available_lock, m_result_available_lock;
  bool m_should_stop_perform_thread;
  bool m_warmup;

private:
  enum UGenInputs { modelIdx=0, methodIdx, bufSize, warmup, debug, inputs };
  void clearOutputs(int nSamples);
  bool allocBuffers();
  void updateAttributes();

  RingBuf* m_inBuffer;
  RingBuf* m_outBuffer;
  int16 m_ugenId;
  bool m_useThread;

  std::unique_ptr<std::thread> m_compute_thread;
};

} // namespace NN
