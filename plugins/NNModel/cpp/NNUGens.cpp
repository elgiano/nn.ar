// NNUGens.cpp

#include "NNUGens.hpp"
#include "NNModel.hpp"
#include "NNModelCmd.hpp"
#include "rt_circular_buffer.h"
#include "backend.h"
#include "SC_InterfaceTable.h"
#include "SC_PlugIn.hpp"
#include <cstdio>
#include <fstream>
#include <ostream>

static InterfaceTable* ft;

namespace NN {


NNModel::NNModel(): m_backend(), m_methods() {
}

NNModel* NNModelRegistry::get(std::string key, bool warn) {
  auto model = models[key];
  if (model == nullptr) {
    if (warn) Print("NNBackend: %s not found\n", key.c_str());
    return nullptr;
  }
  if (!model->is_loaded()) {
    if (warn) Print("NNBackend: %s not loaded yet\n", key.c_str());
  }
  return model;
}

NNModel* NNModelRegistry::get(unsigned short idx, bool warn) {
  try {
    auto model = modelsByIdx.at(idx);
    if (!model->is_loaded()) {
      if (warn) Print("NNBackend: idx %d not loaded yet\n", idx);
    }
    return model;
  }
  catch (const std::out_of_range&) {
    if (warn) Print("NNBackend: idx %d not found\n", idx);
    return nullptr;
  }
}
void NNModelRegistry::streamAllInfo(std::ostream& dest) {
  for (const auto kv: models) {
    kv.second->streamInfo(dest);
  }
}
void NNModelRegistry::printAllInfo() {
  streamAllInfo(std::cout);
  std::cout << std::endl;
}

bool NNModelRegistry::load(std::string key, const char* path) {
  auto model = get(key, false);
  if (model != nullptr) {
    if (model->m_path == path) {
      Print("NNModel[%s]: already loaded %s\n", key.c_str(), path);
      return true;
    } else {
      return model->load(path);
    }
  }

  model = new NNModel();
  models[key] = model;
  model->m_idx = modelCount++;
  modelsByIdx.push_back(model);
  return model->load(path);
}

bool NNModel::load(const char* path) {
  Print("NNBackend: loading %s\n", path);
  bool loaded = m_backend.load(path) == 0;
  if (loaded) {
    Print("NNBackend: loaded %s\n", path);
  } else {
    Print("ERROR: NNBackend can't load model %s\n", path);
    return false;
  }

  // cache path
  m_path = path;

  // cache methods
  if (m_methods.size() > 0) m_methods.clear();
  for (std::string name: m_backend.get_available_methods()) {
    auto params = m_backend.get_method_params(name);
    // skip methods with no params
    if (params.size() == 0) continue;
    NNModelMethod m(name, params);
    m_methods.push_back(m);
  }
  m_higherRatio = m_backend.get_higher_ratio();

  // cache settings
  if (m_settings.size() > 0) m_settings.clear();
  for (std::string name: m_backend.get_settable_attributes()) {
    m_settings.push_back(name);
  }

  return true;
}

bool NNModel::set(std::string name, std::string value, bool warn) {
  /* Print("setting %s to %f\n", name.c_str(), value); */
  std::vector<std::string> args = {value};
  try {
    m_backend.set_attribute(name, args);
  } catch (...) {
    if (warn) Print("NNBackend: can't set attribute %s\n", name.c_str());
    return false;
  }
  return true;
}
bool NNModel::set(unsigned short settingIdx, std::string value, bool warn) {
  std::string setting = getSetting(settingIdx, warn);
  if (setting.empty()) return false;
  return set(setting, value, warn);
};
bool NNModel::set(unsigned short settingIdx, float value, bool warn) {
  return set(settingIdx, std::to_string(value), warn);
}

float NNModel::get(std::string name, bool warn) {
  /* Print("setting %s to %f\n", name.c_str(), value); */
  try {
    auto value = m_backend.get_attribute(name)[0];
    /* auto str = m_backend.get_attribute_as_string(name); */
    /* Print("STR %s\n", str.c_str()); */
    if (value.isInt()) {
      return static_cast<float>(value.toInt());
    }
    else if (value.isBool()) {
      return value.toBool() ? 1.0 : 0.0;
    }
    else if (value.isDouble()) {
      return static_cast<float>(value.toDouble());
    }
    else {
      if (warn) Print("NNBackend: attribute '%s' has unsupported type.\n", name.c_str());
      return 0;
    }
  } catch (...) {
    if (warn) Print("NNBackend: can't get attribute %s\n", name.c_str());
    return 0;
  }
}
float NNModel::get(unsigned short settingIdx, bool warn) {
  std::string setting = getSetting(settingIdx, warn);
  if (setting.empty()) return false;
  return get(setting, warn);
};


NNModelMethod::NNModelMethod(std::string name, const std::vector<int>& params):
name(name) {
  inDim = params[0];
  inRatio = params[1];
  outDim = params[2];
  outRatio = params[3];
}

NNModelMethod* NNModel::getMethod(unsigned short idx, bool warn) {
  try {
    return &m_methods.at(idx);
  } catch (const std::out_of_range&) {
    if (warn) Print("NNBackend: method %d not found\n", idx);
    return nullptr;
  }
}

std::string NNModel::getSetting(unsigned short idx, bool warn) {
  try {
    return m_settings.at(idx);
  } catch (const std::out_of_range&) {
    if (warn) Print("NNBackend: setting %d not found\n", idx);
    return nullptr;
  }
}

// file dumps are needed to share info with client

void NNModel::streamInfo(std::ostream& stream) {
  stream << "- idx: " << m_idx
    << "\n  modelPath: " << m_path
    << "\n  minBufferSize: " << m_higherRatio
    << "\n  methods:";
  for (auto m: m_methods) {
    stream << "\n    - name: " << m.name
      << "\n      inDim: " << m.inDim
      << "\n      inRatio: " << m.inRatio
      << "\n      outDim: " << m.outDim
      << "\n      outRatio: " << m.outRatio;
  }
  auto settings = m_backend.get_settable_attributes();
  if (settings.size() > 0) {
    stream << "\n  settings:";
    for(std::string attr: settings)
      stream << "\n    - " << attr; 
  }
  stream << "\n";
}

void NNModel::printInfo() {
  streamInfo(std::cout);
  std::cout << std::endl;
}

bool NNModel::dumpInfo(const char* filename) {
  std::ofstream file;
  try {
    file.open(filename);
    if (!file.is_open()) {
      Print("ERROR: NNBackend couldn't open file %s\n", filename);
      return false;
    }
    streamInfo(file);
    file.close();
    return true;
  }
  catch (...) {
    Print("ERROR: NNBackend couldn't dump info to file %s\n", filename);
    return false;
  }
}

// UGEN

NNModel* getModel(float modelIdx) {
  auto model = gModels.get(static_cast<unsigned short>(modelIdx));
  if (model == nullptr)
    Print("NNBackend: model %d not found");
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

NN::NN(): 
  m_model(nullptr), m_method(nullptr), 
  m_compute_thread(nullptr), m_useThread(true),
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
    Print("NNBackend: blockSize(%d) larger than model bufferSize(%d), disabling\n", bufferSize(), m_bufferSize);
    mDone = true;
    set_calc_function<NN, &NN::clearOutputs>();
    return;
  }

  /* Print("NN: Ctor done\n"); */
  m_enabled = true;
  mCalcFunc = make_calc_function<NN, &NN::next>();
  clearOutputs(1);
}

NN::~NN() {
  /* Print("NN: Dtor\n"); */
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
  if (m_bufferSize < 0) {
    // NO THREAD MODE
    m_useThread = false;
    m_bufferSize = m_model->m_higherRatio;
  } else if (m_bufferSize < m_model->m_higherRatio) {
    m_bufferSize = m_model->m_higherRatio;
    /* Print("NN: buffer size to small, switching to %d.\n", m_bufferSize); */
  } else {
    m_bufferSize = NEXTPOWEROFTWO(m_bufferSize);
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
  if (m_compute_thread) m_compute_thread->join();
  /* Print("NN: freeing buffers\n"); */
  RTFree(mWorld, m_inModel);
  RTFree(mWorld, m_outModel);
  freeRingBuffer(mWorld, m_inBuffer);
  freeRingBuffer(mWorld, m_outBuffer);
}

// PERFORM

void model_perform(NN* nn_instance) {
  /* Print("NN: performing\n"); */
  auto m_bufferSize = nn_instance->m_bufferSize;
  std::vector<float*> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; c++)
    in_model.push_back(&nn_instance->m_inModel[c * m_bufferSize]);
  for (int c(0); c < nn_instance->m_outDim; c++)
    out_model.push_back(&nn_instance->m_outModel[c * m_bufferSize]);
  nn_instance->m_model->perform(in_model, out_model, nn_instance->m_bufferSize,
                                nn_instance->m_method->name, 1);
}

void NN::next(int nSamples) {
  if (!m_model->is_loaded() || mDone || !m_enabled) {
    ClearUnitOutputs(this, nSamples);
    return;
  };

  // copy inputs to circular buffer
  for (int c(0); c < m_inDim; ++c) {
    const float* samples = in(UGenInputs::inputs + c);
    m_inBuffer[c].put(samples, bufferSize());
  }

  if (m_inBuffer[0].full()) {

    if (m_useThread && m_compute_thread)
      m_compute_thread->join();

    // transfer samples from inBuffer to model inBuf
    for (int c(0); c < m_inDim; ++c)
      m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);

    auto a = &m_inModel[1];

    if(!m_useThread)
      model_perform(this);

    // transfer samples from model outBuf to outBuffer
    for (int c(0); c < m_outDim; ++c)
      m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);

    if(m_useThread)
      m_compute_thread = std::make_unique<std::thread>(model_perform, this);
  }

  // copy circular buf to out
  for (int c(0); c < m_outDim; ++c) {
    float* samples = out(c);
    m_outBuffer[c].get(samples, bufferSize());
  }
}

// PARAMS

NNParamUGen::NNParamUGen(): m_model(nullptr) {
  m_model = getModel(in0(NNParamInputs::modelIdx));
  /* Print("NNParamUGen: Ctor\nmodel: %p\n", m_model); */
  m_setting = static_cast<unsigned short>(in0(NNParamInputs::settingIdx));
  /* Print("setting: #%d\n", m_setting); */
  std::string setting;
  if (m_model != nullptr)
    setting = m_model->getSetting(m_setting);
  if (setting.empty()) {
    mDone = true;
    Unit* unit = this;
    SETCALC(ClearUnitOutputs);
    return;
  }
  /* Print("settingName: %s\n", setting.c_str()); */
}

NNSet::NNSet() {
  set_calc_function<NNSet, &NNSet::next>();
}
void NNSet::next(int nSamples) {
  Unit* unit = this;
  ClearUnitOutputs;
  if (mDone) { return; }
  m_model->set(m_setting, in0(UGenInputs::value));
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
  out0(0) = m_model->get(m_setting, true);
}

} // namespace NN

void nrtFree(World*, void* data) { NRTFree(data); }

template<class CmdData, auto cmdFn>
void asyncCmd(World* world, void* inUserData, sc_msg_iter* args, void* replyAddr) {
  const char* cmdName = ""; // used only in /done, we use /sync instead
  CmdData* data = CmdData::alloc(args, ft, nullptr);
  if (data == nullptr) return;
  DoAsynchronousCommand(
    world, replyAddr, "", data,
    cmdFn, // stage2 is non real time
    nullptr, // stage3: RT (completion msg performed if true)
    nullptr, // stage4: NRT (sends /done if true)
    nrtFree, 0, 0);
}

PluginLoad(NNUGens) {
  // Plugin magic
  ft = inTable;

  DefinePlugInCmd("/nn_load", asyncCmd<NN::LoadCmdData, NN::doLoadMsg>, nullptr);
  DefinePlugInCmd("/nn_query", asyncCmd<NN::QueryCmdData, NN::doQueryMsg>, nullptr);
  DefinePlugInCmd("/nn_set", asyncCmd<NN::SetCmdData, NN::doSetMsg>, nullptr);
  registerUnit<NN::NN>(ft, "NNUGen", false);
  registerUnit<NN::NNSet>(ft, "NNSet", false);
  registerUnit<NN::NNGet>(ft, "NNGet", false);
}
