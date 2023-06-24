// NNUGens.cpp
#include "NNModel.hpp"
#include "NNUGens.hpp"
#include "NNModelCmd.hpp"
#include "rt_circular_buffer.h"
#include "SC_InterfaceTable.h"
#include "SC_PlugIn.hpp"

InterfaceTable* ft;

// global model store, by numeric id
NN::NNModelRegistry gModels;

namespace NN {

NNModel* getModel(float modelIdx) {
  auto model = gModels.get(static_cast<unsigned short>(modelIdx), true);
  if (model == nullptr)
    Print("NNUGen: model %d not found\n");
  return model;
}

NNModelMethod* getModelMethod(NNModel* model, float methodIdx) {
  if (model == nullptr) return nullptr;

  auto method = model->getMethod(static_cast<unsigned short>(methodIdx));
  if (method == nullptr)
    Print("NNBackend: method %d not found\n", methodIdx);
  return method;
}

void NN::clearOutputs(int nSamples) {
  ClearUnitOutputs(this, nSamples);
}


// PERFORM

#define MODEL_PERFORM_CUSTOM 0

#if MODEL_PERFORM_CUSTOM

void model_perform(NN* nn_instance) {
  nn_instance->m_model->perform(
        nn_instance->m_inModel, nn_instance->m_outModel, nn_instance->m_bufferSize,
        nn_instance->m_method, 1);
}
void model_perform_loop(NN* nn_instance) {
  while (!nn_instance->m_should_stop_perform_thread) {
    if (nn_instance->m_data_available_lock.try_acquire_for(
            std::chrono::milliseconds(200))) {
      model_perform(nn_instance);
      nn_instance->m_result_available_lock.release();
    }
  }
}

#else

void model_perform(NN* nn_instance) {
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < nn_instance->m_outDim; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
      nn_instance->m_model->perform(in_model, out_model,
                                    nn_instance->m_bufferSize,
                                    nn_instance->m_method->name, 1);
}
void model_perform_loop(NN *nn_instance) {
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < nn_instance->m_outDim; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
  while (!nn_instance->m_should_stop_perform_thread) {
    if (nn_instance->m_data_available_lock.try_acquire_for(
            std::chrono::milliseconds(200))) {
      nn_instance->m_model->perform(in_model, out_model,
                                    nn_instance->m_bufferSize,
                                    nn_instance->m_method->name, 1);
      nn_instance->m_result_available_lock.release();
    }
  }
}

#endif // def MODEL_PERFORM_CUSTOM

void NN::next(int nSamples) {
  if (!m_model->is_loaded() || mDone || !m_enabled) {
    ClearUnitOutputs(this, nSamples);
    return;
  };

  // copy inputs to circular buffer
  for (int c(0); c < m_inDim; ++c) {
    m_inBuffer[c].put(in(UGenInputs::inputs + c), bufferSize());
  }

  if (m_inBuffer[0].full()) {

    if (!m_useThread) {

      for (int c(0); c < m_inDim; ++c)
        m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);

      model_perform(this);

      for (int c(0); c < m_outDim; ++c)
        m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);
    } else if (m_result_available_lock.try_acquire()) {
      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_inDim; ++c)
        m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);
      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_outDim; ++c)
        m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);
      // SIGNAL PERFORM THREAD THAT DATA IS AVAILABLE
      m_data_available_lock.release();
    }
  }

  // copy circular buf to out
  for (int c(0); c < m_outDim; ++c)
    m_outBuffer[c].get(out(c), bufferSize());
}

NN::NN(): 
  m_model(nullptr), m_method(nullptr), 
  m_compute_thread(nullptr), m_useThread(true),
  m_should_stop_perform_thread(false),
  m_data_available_lock(0), m_result_available_lock(1),
  m_enabled(false),
  m_inBuffer(nullptr), m_outBuffer(nullptr),
  m_inModel(nullptr), m_outModel(nullptr)
{
  /* Print("NN: Ctor\n"); */
  m_model = getModel(in0(UGenInputs::modelIdx));
  if (m_model)
    m_method = getModelMethod(m_model, in0(UGenInputs::methodIdx));
  if (m_method == nullptr) {
    mDone = true;
    set_calc_function<NN, &NN::clearOutputs>();
    return;
  }
  m_inDim = m_method->inDim;
  m_outDim = m_method->outDim;

  if (!allocBuffers()) {
    freeBuffers();
    Unit* unit = this;
    ClearUnitOnMemFailed;
  }

  if (bufferSize() > m_bufferSize) {
    Print("NNUGen: blockSize(%d) larger than model bufferSize(%d), disabling\n", bufferSize(), m_bufferSize);
    mDone = true;
    set_calc_function<NN, &NN::clearOutputs>();
    return;
  }

  /* Print("NN: Ctor done\n"); */
  // don't use external thread on NRT
  if (!mWorld->mRealTime) m_useThread = false;
  if (m_useThread)
    m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
  mCalcFunc = make_calc_function<NN, &NN::next>();
  m_enabled = true;
}

NN::~NN() {
  /* Print("NN: Dtor\n"); */
  m_should_stop_perform_thread = true;
  if (m_compute_thread) m_compute_thread->join();
  freeBuffers();
}

// BUFFERS

template<class T>
T* rtAlloc(World* world, size_t size) {
  return (T*) RTAlloc(world, sizeof(T) * size);
}

RingBuf* allocRingBuffer(World* world, size_t bufSize, size_t numChannels) {
  RingBuf* ctrs = rtAlloc<RingBuf>(world, numChannels);
  float* data = rtAlloc<float>(world, numChannels * bufSize);
  if (ctrs == nullptr || data == nullptr) {
    RTFree(world, ctrs); return nullptr;
  };
  memset(data, 0, sizeof(float) * numChannels * bufSize);
  for (int c(0); c < numChannels; ++c) {
    auto buf = data + (bufSize * c);
    new(ctrs + c) RingBuf(buf, bufSize);
  };
  /* Print("ctrs: %p\ndata: %p\n", ctrs, data); */
  return ctrs;
}

void freeRingBuffer(World* world, RingBuf* buf) {
  if (buf == nullptr) return;
  RTFree(world, buf[0].getBuffer()); // data
  RTFree(world, buf);
}

bool NN::allocBuffers() {

  m_bufferSize = in0(UGenInputs::bufSize);
  if (m_bufferSize <= 0) {
    // NO THREAD MODE
    m_useThread = false;
    m_bufferSize = m_model->m_higherRatio;
  } else if (m_bufferSize < m_model->m_higherRatio) {
    m_bufferSize = m_model->m_higherRatio;
    Print("NNUGen: buffer size to small, switching to %d.\n", m_bufferSize);
  } else {
    m_bufferSize = NEXTPOWEROFTWO(m_bufferSize);
    Print("NNUGen: rounding buffer size %d.\n", m_bufferSize);
  }

  m_inBuffer = allocRingBuffer(mWorld, m_bufferSize, m_inDim);
  if (m_inBuffer == nullptr) return false;
  m_outBuffer = allocRingBuffer(mWorld, m_bufferSize, m_outDim);
  if (m_outBuffer == nullptr) return false;
  m_inModel = rtAlloc<float>(mWorld, m_bufferSize * m_inDim);
  if(m_inModel == nullptr) return false;
  m_outModel = rtAlloc<float>(mWorld, m_bufferSize * m_outDim);
  if(m_outModel == nullptr) return false;
  memset(m_inModel, 0, sizeof(float) * m_bufferSize * m_inDim);
  memset(m_outModel, 0, sizeof(float) * m_bufferSize * m_outDim);
  /* Print("m_inModel: %p\nm_outModel: %p\n", m_inModel, m_outModel); */
  return true;
}

void NN::freeBuffers() {
  /* Print("NN: freeing buffers\n"); */
  RTFree(mWorld, m_inModel);
  RTFree(mWorld, m_outModel);
  freeRingBuffer(mWorld, m_inBuffer);
  freeRingBuffer(mWorld, m_outBuffer);
}


// PARAMS

NNAttrUGen::NNAttrUGen(): m_model(nullptr) {
  m_model = getModel(in0(NNAttrInputs::modelIdx));
  /* Print("NNAttrUGen: Ctor\nmodel: %p\n", m_model); */
  m_attrIdx = static_cast<unsigned short>(in0(NNAttrInputs::attrIdx));
  /* Print("attr: #%d\n", m_attrName); */
  std::string attrName;
  if (m_model != nullptr)
    attrName = m_model->getAttribute(m_attrIdx);
  if (attrName.empty()) {
    mDone = true;
    Unit* unit = this;
    SETCALC(ClearUnitOutputs);
    return;
  }
  /* Print("attrName: %s\n", attrName.c_str()); */
}

NNSet::NNSet() {
  set_calc_function<NNSet, &NNSet::next>();
}
void NNSet::next(int nSamples) {
  Unit* unit = this;
  ClearUnitOutputs;
  if (mDone) { return; }
  m_model->set(m_attrIdx, in0(UGenInputs::value));
}

NNGet::NNGet() {
  set_calc_function<NNGet, &NNGet::next>();
}
void NNGet::next(int nSamples) {
  Unit* unit = this;
  ClearUnitOutputs;
  if (mDone) {
    return;
  };
  out0(0) = m_model->get(m_attrIdx, true);
}


} // namespace NN


PluginLoad(NNUGens) {
  // Plugin magic
  ft = inTable;

  DefinePlugInCmd("/nn_load", NN::Cmd::cmd_nn_load, nullptr);
  DefinePlugInCmd("/nn_set", NN::Cmd::cmd_nn_set, nullptr);
  DefinePlugInCmd("/nn_query", NN::Cmd::cmd_nn_query, nullptr);
  registerUnit<NN::NN>(ft, "NNUGen", false);
  registerUnit<NN::NNSet>(ft, "NNSet", false);
  registerUnit<NN::NNGet>(ft, "NNGet", false);
}
