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

struct NNID { int32 parentID; int32 nodeID; };
inline bool operator<(const NNID& p1, const NNID& p2) {
  return (p1.parentID != p2.parentID) ? 
        p1.parentID < p2.parentID :
        p1.nodeID < p2.nodeID;
}

class NNUGenRegister {
public:
  void add(int32 id, NN* ugen) {
    Print("registering ugen: #(%d, %d) %p\n", ugen->mParent->mNode.mHash, id, ugen);
    m_ugens[NNID{ugen->mParent->mNode.mHash, id}] = ugen;
  }
  void remove(int32 id, const NN* ugen) {
    try {
      m_ugens.erase(NNID{ugen->mParent->mNode.mHash, id});
    } catch (const std::out_of_range&) { }
  }
  NN* get(int32 id, const Unit* unit) {
    Print("getting ugen: #(%d, %d)\n", unit->mParent->mNode.mHash, id);
    try {
      return m_ugens.at(NNID{unit->mParent->mNode.mHash, id});
    } catch (const std::out_of_range&) {
      Print("NNUGen #(%d, %d) not found\n", unit->mParent->mNode.mHash, id);
      return nullptr;
    }
  }
private:
  std::map<NNID, NN*> m_ugens;
};

// global UGen reference store, for setting and getting attributes
static NNUGenRegister nnUGenRegister;


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


// PERFORM

void model_perform_load(NN* nn) {
  Print("loading\n");
  nn->m_model.load(nn->m_modelDesc->m_path);
  Print("loaded\n");
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
      if(!nn_instance->m_should_stop_perform_thread)
        nn_instance->m_model.perform(in_model, out_model,
                                    nn_instance->m_bufferSize,
                                    nn_instance->m_method->name, 1);

      nn_instance->m_result_available_lock.release();
    }
  }
  /* Print("thread exit\n"); */
}

#endif // def MODEL_PERFORM_CUSTOM


void NN::next(int nSamples) {

  if (!m_modelDesc->is_loaded() || mDone || !m_enabled) {
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
  m_method(nullptr), 
  m_compute_thread(nullptr), m_useThread(true),
  m_data_available_lock(0), m_result_available_lock(1),
  m_should_stop_perform_thread(false),
  m_enabled(false),
  m_inBuffer(nullptr), m_outBuffer(nullptr),
  m_inModel(nullptr), m_outModel(nullptr)
{
  int32 m_ugenID = static_cast<int32>(in0(UGenInputs::ugenIdx));
  nnUGenRegister.add(m_ugenID, this);
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
    m_compute_thread = new std::thread(model_perform_loop, this);
  else
    model_perform_load(this);

  mCalcFunc = make_calc_function<NN, &NN::next>();
  m_enabled = true;
  /* Print("NN: Ctor done\n"); */
}

NN::~NN() {
  /* Print("NN: Dtor\n"); */
  nnUGenRegister.remove(m_ugenId, this);
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


// PARAMS





NNAttrUGen::NNAttrUGen():m_attrName() {

  /* Print("NNATTR::CTor\n"); */
  int32 ugenIdx = static_cast<int32>(in0(NNAttrInputs::modelIdx));
  m_ugen = nnUGenRegister.get(ugenIdx, this);
  /* Print("> ugen: #%d %p\n", ugenIdx, m_ugen); */
  auto attrIdx = static_cast<unsigned short>(in0(NNAttrInputs::attrIdx));
  if (m_ugen != nullptr)
    m_attrName = m_ugen->m_modelDesc->getAttributeName(attrIdx);
  /* Print("> attr #%d: %s\n", attrIdx, m_attrName.c_str()); */
  if (m_attrName.empty()) {
    mDone = true;
    Unit* unit = this;
    SETCALC(ClearUnitOutputs);
    return;
  }
}



NNSet::NNSet() {
  set_calc_function<NNSet, &NNSet::next>();
}
static bool setAttribute(Backend& backend, const std::string& attrName, float value, bool warn=false) {
/*   /1* Print("setting attr %s to %f\n", name.c_str(), value); *1/ */
  std::vector<std::string> args = {std::to_string(value)};
  try {
    backend.set_attribute(attrName, args);
  } catch (...) {
    if (warn) Print("NNBackend: can't set attribute %s\n", attrName.c_str());
    return false;
  }
  return true;
}
void NNSet::next(int nSamples) {
  Unit* unit = this;
  ClearUnitOutputs;
  if (mDone) { return; };
  float val = in0(UGenInputs::value);
  if (!m_init || val != m_lastVal) {
    setAttribute(m_ugen->m_model, m_attrName, val);
    m_lastVal = val;
    m_init = true;
  }
}

NNGet::NNGet() {
  set_calc_function<NNGet, &NNGet::next>();
}
static float getAttribute(Backend& backend, const std::string& name, bool warn) {
  try {
    auto value = backend.get_attribute(name)[0];
    /* auto str = m_backend.get_attribute_as_string(name); */
    /* Print("STR %s\n", str.c_str()); */
    if (value.isInt()) {
      return static_cast<float>(value.toInt());
    }
    else if (value.isBool()) {
      return value.toBool() ? 1.0 : 0.0;
    }
    else if (value.isDouble()) {
      Print("%s: %d\n", name.c_str(), value.toDouble());
      return static_cast<float>(value.toDouble());
    }
    else {
      if (warn) Print("NNGet: attribute '%s' has unsupported type.\n", name.c_str());
      return 0;
    }
  } catch (...) {
    if (warn) Print("NNGet: can't get attribute %s\n", name.c_str());
    return 0;
  }
}

void NNGet::next(int nSamples) {
  Unit* unit = this;
  ClearUnitOutputs;
  if (mDone) {
    return;
  };
  out0(0) = getAttribute(m_ugen->m_model, m_attrName, true);
}

} // namespace NN


PluginLoad(NNUGens) {
  // Plugin magic
  ft = inTable;

  registerUnit<NN::NN>(ft, "NNUGen", false);
  registerUnit<NN::NNSet>(ft, "NNSet", false);
  registerUnit<NN::NNGet>(ft, "NNGet", false);
  NN::Cmd::definePlugInCmds();
}
