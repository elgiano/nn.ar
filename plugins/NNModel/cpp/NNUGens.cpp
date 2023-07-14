// NNUGens.cpp
#include "NNModel.hpp"
#include "NNUGens.hpp"
#include "NNModelCmd.hpp"
#include "SC_Unit.h"
#include "rt_circular_buffer.h"
#include "SC_InterfaceTable.h"
#include "SC_PlugIn.hpp"

InterfaceTable* ft;

// global model store, by numeric id
NN::NNModelDescLib gModels;


template<class T>
T* rtAlloc(World* world, size_t size=1) {
  return (T*) RTAlloc(world, sizeof(T) * size);
}

namespace NN {

static NNModelMethod* getModelMethod(NNModelDesc* model, float methodIdx) {
  if (model == nullptr) return nullptr;

  auto method = model->getMethod(static_cast<unsigned short>(methodIdx));
  if (method == nullptr)
    Print("NNBackend: method %d not found\n", methodIdx);
  return method;
}

void NN::clearOutputs(int nSamples) {
  ClearUnitOutputs(this, nSamples);
}

// ATTRIBUTES
NNSetAttr::NNSetAttr(std::string name, int inputIdx, float initVal):
    attrName(name), inputIdx(inputIdx), value(initVal), valUpdated(true) {}

void NNSetAttr::update(Unit* unit, int nSamples) {
  float newval = IN0(inputIdx);
  if (newval != value) {
    value = newval;
    valUpdated = true;
  }
}

// attributes are provided as additional input pairs (attrId, val) after model inputs
void NN::setupAttributes() {
  int i = UGenInputs::inputs + m_inDim;
  while (i < numInputs()) {
    int attrIdx = in0(i);
    std::string attrName = m_modelDesc->getAttributeName(attrIdx, true);
    if (!attrName.empty()) {
      int inputIdx = i + 1;
      NNSetAttr setter(attrName, inputIdx, in0(inputIdx));
      m_attributes.push_back(setter);
    } else {
      Print("NNUGen: attribute #%d not found\n", attrIdx);
    }
    i += 2; // attrIdx, val
  }
}

static void model_perform_attributes(NN* nn_instance) {
  for(auto& a: nn_instance->m_attributes) {
    if(!a.changed()) continue;
    try {
      nn_instance->m_model.set_attribute(a.attrName, {a.getStrValue()});
      // print attr value if debugging
      if (nn_instance->m_debug >= Debug::attributes) {
        auto currVal = nn_instance->m_model.get_attribute_as_string(a.attrName);
        Print("%s: %s\n", a.attrName.c_str(), currVal.c_str());
      }
    } catch (...) {
      Print("NNUGen: can't set attribute %s\n", a.attrName.c_str());
    }
  };
}

// PERFORM

void model_perform_load(NN* nn) {
  auto path = nn->m_modelDesc->m_path;
  if (nn->m_debug >= Debug::all)
    Print("NNUGen: loading model %s\n", path.c_str());
  int err = nn->m_model.load(nn->m_modelDesc->m_path);
  if (err) {
    Print("NNUGen: ERROR loading model %s\n", path.c_str());
    nn->mDone = true;
    return;
  }
  if(nn->m_warmup) {
    if (nn->m_debug >= Debug::all)
      Print("NNUGen: warming up model\n", path.c_str());
    nn->warmupModel();
  }
  nn->setupAttributes();
  if (nn->m_debug >= Debug::all)
    Print("NNUGen: loaded %s\n", path.c_str());
}

#define MODEL_PERFORM_CUSTOM 0

#if MODEL_PERFORM_CUSTOM

/* void model_perform(NN* nn_instance) { */
/*   nn_instance->m_model->perform( */
/*         nn_instance->m_inModel, nn_instance->m_outModel, nn_instance->m_bufferSize, */
/*         nn_instance->m_method, 1); */
/* } */
/* void model_perform_loop(NN* nn_instance) { */
/*   model_perform_load(nn_instance); */
/*   while (!nn_instance->m_should_stop_perform_thread) { */
/*     if (nn_instance->m_data_available_lock.try_acquire_for( */
/*             std::chrono::milliseconds(200))) { */
/*       model_perform(nn_instance); */
/*       nn_instance->m_result_available_lock.release(); */
/*     } */
/*   } */
/*   nn_instance->freeBuffers(); */
/* } */

#else

void model_perform(NN* nn_instance) {
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < nn_instance->m_outDim; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
  model_perform_attributes(nn_instance);
  nn_instance->m_model.perform(in_model, out_model,
                               nn_instance->m_bufferSize,
                               nn_instance->m_method->name, 1);
}
void model_perform_loop(NN *nn_instance) {
  model_perform_load(nn_instance);
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < nn_instance->m_outDim; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
  while (!nn_instance->m_should_stop_perform_thread) {
    if (nn_instance->m_data_available_lock.try_acquire_for(
      std::chrono::milliseconds(200))) {
      if(!nn_instance->m_should_stop_perform_thread) {
        model_perform_attributes(nn_instance);
        nn_instance->m_model.perform(in_model, out_model,
                                     nn_instance->m_bufferSize,
                                     nn_instance->m_method->name, 1);
      }

      nn_instance->m_result_available_lock.release();
    }
  }
  /* Print("thread exit\n"); */
}

#endif // def MODEL_PERFORM_CUSTOM


void NN::next(int nSamples) {

  if (!m_modelDesc->is_loaded() || mDone) {
    ClearUnitOutputs(this, nSamples);
    return;
  };

  // update attr setters
  for (auto& a: m_attributes) a.update(this, nSamples);

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
  m_method(nullptr), 
  m_compute_thread(nullptr), m_useThread(true),
  m_data_available_lock(0), m_result_available_lock(1),
  m_should_stop_perform_thread(false),
  m_inBuffer(nullptr), m_outBuffer(nullptr),
  m_inModel(nullptr), m_outModel(nullptr)
{

  m_debug = static_cast<int>(in0(UGenInputs::debug));
  m_warmup = in0(UGenInputs::warmup) > 0;

  auto modelIdx = static_cast<unsigned short>(in0(UGenInputs::modelIdx));
  m_modelDesc = gModels.get(modelIdx);
  if (m_modelDesc)
    m_method = getModelMethod(m_modelDesc, in0(UGenInputs::methodIdx));
  if (m_method == nullptr) {
    mDone = true;
    set_calc_function<NN, &NN::clearOutputs>();
    return;
  }
  m_inDim = m_method->inDim;
  m_outDim = m_method->outDim;

  m_bufferSize = in0(UGenInputs::bufSize);
  if (m_bufferSize < 0) {
    // NO THREAD MODE
    m_useThread = false;
    m_bufferSize = m_modelDesc->m_higherRatio;
  } else if (m_bufferSize == 0) {
    m_bufferSize = m_modelDesc->m_higherRatio;
  } else if (m_bufferSize < m_modelDesc->m_higherRatio) {
    m_bufferSize = m_modelDesc->m_higherRatio;
    Print("NNUGen: buffer size to small, switching to %d.\n", m_bufferSize);
  } else {
    int pow2 = NEXTPOWEROFTWO(m_bufferSize);
    if (m_bufferSize != pow2) {
      m_bufferSize = pow2;
      Print("NNUGen: rounding buffer size %d.\n", m_bufferSize);
    }
  }

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

  // don't use external thread on NRT
  if (!mWorld->mRealTime) m_useThread = false;
  if (m_useThread)
    m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
  else
    model_perform_load(this);

  mCalcFunc = make_calc_function<NN, &NN::next>();
  /* Print("NN: Ctor done\n"); */
}

NN::~NN() {
  /* Print("NN: Dtor\n"); */
  if (m_compute_thread) {
    // don't wait for join, it would stall the dsp chain
    // thread calls freeBuffers() when stopped
    m_should_stop_perform_thread = true;
    m_compute_thread->join();
  }
  freeBuffers();
}

// BUFFERS

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
  freeRingBuffer(mWorld, m_inBuffer);
  freeRingBuffer(mWorld, m_outBuffer);
  RTFree(mWorld, m_inModel);
  RTFree(mWorld, m_outModel);
  /* delete m_model; */
  /* RTFree(mWorld, m_model); */
}

void NN::warmupModel() {

  std::vector<float *> in_model, out_model;
  for (int c(0); c < m_inDim; ++c)
    in_model.push_back(&m_inModel[m_bufferSize * c]);
  for (int c(0); c < m_outDim; ++c)
    out_model.push_back(&m_outModel[m_bufferSize * c]);

  m_model.perform(in_model, out_model, m_bufferSize, m_method->name, 1);
}

} // namespace NN


PluginLoad(NNUGens) {
  // Plugin magic
  ft = inTable;

  registerUnit<NN::NN>(ft, "NNUGen", false);
  NN::Cmd::definePlugInCmds();
}
