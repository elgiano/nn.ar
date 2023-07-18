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
  const NNModelAttribute* attr;
  // remember in0 indices
  int inputIdx;

  NNSetAttr(const NNModelAttribute* attr, int inputIdx, float initVal);

  // called in audio thread: check trig, update value and flag
  void update(Unit* unit, int nSamples);

  bool changed() const { return valUpdated; }
  // called before model_perform
  const char* getName() {
    return attr->name.c_str();
  }
  std::string getStrValue() {
    valUpdated = false;
    if (attr->type == NNAttributeType::typeBool)
      return value > 0 ? "true" : "false";
    else if (attr->type == NNAttributeType::typeInt)
      return std::to_string(static_cast<int>(value));
    return std::to_string(value);
  }

private:
  float lastTrig = 0;
  float value = 0;
  bool valUpdated = false;
};

class NN {
public:
  NN(World* world, const NNModelDesc* modelDesc, const NNModelMethod* modelMethod,
     float* inModel, float* outModel,  RingBuf* m_inBuffer, RingBuf* m_outBuffer,
     int bufferSize, int m_debug);

  ~NN();

  void warmupModel();

  RingBuf* m_inBuffer;
  RingBuf* m_outBuffer;
  float* m_inModel;
  float* m_outModel;
  const NNModelDesc* m_modelDesc;
  const NNModelMethod* m_method;
  World* mWorld;
  std::vector<NNSetAttr> m_attributes;
  Backend m_model;
  int m_inDim, m_outDim;
  int m_bufferSize, m_debug;
  std::thread* m_compute_thread;
  std::binary_semaphore m_data_available_lock, m_result_available_lock;
  bool m_should_stop_perform_thread;
  bool m_warmup;
};

class NNUGen : public SCUnit {
public:

  NNUGen();
  ~NNUGen();

  void next(int nSamples);
  void freeBuffers();
  void setupAttributes();

  NN* m_sharedData;

private:
  enum UGenInputs { modelIdx=0, methodIdx, bufSize, warmup, debug, inputs };
  void clearOutputs(int nSamples);
  bool allocBuffers();
  void updateAttributes();

  RingBuf* m_inBuffer;
  RingBuf* m_outBuffer;
  float* m_inModel;
  float* m_outModel;
  int m_inDim, m_outDim;
  int m_bufferSize, m_debug;
  bool m_useThread;
};

} // namespace NN
