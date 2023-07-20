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

static const NNModelMethod* getModelMethod(const NNModelDesc* model, float methodIdx) {
  if (model == nullptr) return nullptr;

  auto method = model->getMethod(static_cast<unsigned short>(methodIdx));
  if (method == nullptr)
    Print("NNBackend: method %d not found\n", methodIdx);
  return method;
}

void NNUGen::clearOutputs(int nSamples) {
  ClearUnitOutputs(this, nSamples);
}

// ATTRIBUTES
NNSetAttr::NNSetAttr(const NNModelAttribute* attr, int inputIdx, float initVal):
    attr(attr), inputIdx(inputIdx), value(initVal), valUpdated(true) {}

void NNSetAttr::update(Unit* unit, int nSamples) {
  float newval = IN0(inputIdx);
  if (newval != value) {
    value = newval;
    valUpdated = true;
  }
}

// attributes are provided as additional input pairs (attrId, val) after model inputs
void NNUGen::setupAttributes() {
  int i = UGenInputs::inputs + m_inDim;
  while (i < numInputs()) {
    int attrIdx = in0(i);
    auto attr = m_sharedData->m_modelDesc->getAttribute(attrIdx, true);
    if (attr != nullptr) {
      int inputIdx = i + 1;
      NNSetAttr setter(attr, inputIdx, in0(inputIdx));
      m_sharedData->m_attributes.push_back(setter);
    } else {
      Print("NNUGen: attribute #%d not found\n", attrIdx);
    }
    i += 2; // attrIdx, val
  }
}

static void model_perform_attributes(NN* nn_instance) {
  for(auto& attr: nn_instance->m_attributes) {
    if(!attr.changed()) continue;
    const char* attrName = attr.getName();
    try {
      nn_instance->m_model.set_attribute(attrName, {attr.getStrValue()});
      // print attr value if debugging
      if (nn_instance->m_debug >= Debug::attributes) {
        auto currVal = nn_instance->m_model.get_attribute_as_string(attrName);
        Print("%s: %s\n", attrName, currVal.c_str());
      }
    } catch (...) {
      Print("NNUGen: can't set attribute %s\n", attrName);
    }
  };
}

// PERFORM

void model_perform_load(NN* nn, bool warmup) {
  auto path = nn->m_modelDesc->getPath();
  if (nn->m_debug >= Debug::all)
    Print("NNUGen: loading model %s\n", path);
  int err = nn->m_model.load(path);
  if (err) {
    Print("NNUGen: ERROR loading model %s\n", path);
    return;
  }
  if(warmup) {
    if (nn->m_debug >= Debug::all)
      Print("NNUGen: warming up model\n", path);
    nn->warmupModel();
  }
  nn->m_loaded = true;
  if (nn->m_debug >= Debug::all)
    Print("NNUGen: loaded %s\n", path);
}

void model_perform_cleanup(NN* nn_instance) {
  auto mWorld = nn_instance->mWorld;
  // manually call destructor and free instance
  nn_instance->~NN();
  RTFree(mWorld, nn_instance);
}

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

void model_perform_loop(NN *nn_instance, bool warmup) {
  model_perform_load(nn_instance, warmup);
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < nn_instance->m_outDim; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
  while (!nn_instance->m_should_stop_perform_thread) {
    if (nn_instance->m_data_available_lock.try_acquire_for(
      std::chrono::milliseconds(200))) {
        model_perform_attributes(nn_instance);
        nn_instance->m_model.perform(in_model, out_model,
                                     nn_instance->m_bufferSize,
                                     nn_instance->m_method->name, 1);
      nn_instance->m_result_available_lock.release();
    }
  }
  model_perform_cleanup(nn_instance);
  /* Print("thread exit\n"); */
}

void NNUGen::next(int nSamples) {

  if (!m_sharedData->m_loaded) {
    ClearUnitOutputs(this, nSamples);
    return;
  };

  // update attr setters
  for (auto& a: m_sharedData->m_attributes) a.update(this, nSamples);

  // copy inputs to circular buffer
  for (int c(0); c < m_inDim; ++c) {
    m_inBuffer[c].put(in(UGenInputs::inputs + c), bufferSize());
  }

  if (m_inBuffer[0].full()) {

    if (!m_useThread) {

      for (int c(0); c < m_inDim; ++c)
        m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);

      model_perform(m_sharedData);

      for (int c(0); c < m_outDim; ++c)
        m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);
    } else if (m_sharedData->m_result_available_lock.try_acquire()) {
      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_inDim; ++c)
        m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);
      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_outDim; ++c)
        m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);
      // SIGNAL PERFORM THREAD THAT DATA IS AVAILABLE
      m_sharedData->m_data_available_lock.release();
    }
  }

  // copy circular buf to out
  for (int c(0); c < m_sharedData->m_outDim; ++c)
    m_outBuffer[c].get(out(c), bufferSize());
}

NN::NN(
  World* world,
  const NNModelDesc* modelDesc, const NNModelMethod* modelMethod,
  float* inModel, float* outModel,  
  RingBuf* inRing, RingBuf* outRing,
  int bufferSize, int debug): 
  mWorld(world),
  m_inModel(inModel), m_outModel(outModel),
  m_inBuffer(inRing), m_outBuffer(outRing),
  m_method(modelMethod), m_modelDesc(modelDesc), 
  m_bufferSize(bufferSize), m_debug(debug),
  m_compute_thread(nullptr),
  m_data_available_lock(0), m_result_available_lock(1),
  m_should_stop_perform_thread(false), m_loaded(false)
{
  m_inDim = m_method->inDim;
  m_outDim = m_method->outDim;
}


NNUGen::NNUGen(): 
  m_inBuffer(nullptr), m_outBuffer(nullptr)
{
  auto modelIdx = static_cast<unsigned short>(in0(UGenInputs::modelIdx));
  const NNModelDesc* modelDesc = gModels.get(modelIdx);
  const NNModelMethod* modelMethod = nullptr;
  if (modelDesc)
    modelMethod = getModelMethod(modelDesc, in0(UGenInputs::methodIdx));
  if (modelMethod == nullptr) {
    set_calc_function<NNUGen, &NNUGen::clearOutputs>();
    return;
  }
  m_inDim = modelMethod->inDim;
  m_outDim = modelMethod->outDim;

  m_bufferSize = in0(UGenInputs::bufSize);

  // don't use external thread on NRT
  m_useThread = mWorld->mRealTime;
  int modelHigherRatio = modelDesc->getHigherRatio();
  if (m_bufferSize < 0) {
    m_bufferSize = modelHigherRatio;
  } else if (m_bufferSize == 0) {
    // NO THREAD MODE
    m_useThread = false;
    m_bufferSize = modelHigherRatio;
  } else if (m_bufferSize < modelHigherRatio) {
    m_bufferSize = modelHigherRatio;
    Print("NNUGen: buffer size to small, switching to %d.\n", m_bufferSize);
  } else {
    int pow2 = NEXTPOWEROFTWO(m_bufferSize);
    if (m_bufferSize != pow2) {
      m_bufferSize = pow2;
      Print("NNUGen: rounding buffer size %d.\n", m_bufferSize);
    }
  }

  if (bufferSize() > m_bufferSize) {
    Print("NNUGen: blockSize(%d) larger than model bufferSize(%d), disabling\n", bufferSize(), m_bufferSize);
    set_calc_function<NNUGen, &NNUGen::clearOutputs>();
    return;
  }

  Unit* unit = this;
  if (!allocBuffers()) {
    freeBuffers();
    ClearUnitOnMemFailed;
  }

  m_debug = static_cast<int>(in0(UGenInputs::debug));

  void* data = RTAlloc(mWorld, sizeof(NN));
  if (!data) {
    freeBuffers();
    ClearUnitOnMemFailed;
  }
  m_sharedData = new(data) NN(mWorld, modelDesc, modelMethod, 
                        m_inModel, m_outModel, m_inBuffer, m_outBuffer,
                        m_bufferSize, m_debug);

  bool warmup = in0(UGenInputs::warmup) > 0;
  if (m_useThread)
    m_sharedData->m_compute_thread = new std::thread(model_perform_loop, m_sharedData, warmup);
  else
    model_perform_load(m_sharedData, warmup);

  setupAttributes();

  mCalcFunc = make_calc_function<NNUGen, &NNUGen::next>();
  /* Print("NN: Ctor done\n"); */
}

NNUGen::~NNUGen() {
  /* Print("NN: Dtor\n"); */
  if (m_sharedData->m_compute_thread) {
    // don't wait for join, it would stall the dsp chain
    // thread frees resources when stopped
    m_sharedData->m_should_stop_perform_thread = true;
    /* m_compute_thread->join(); */
  } else {
    /* Print("freeing manually\n"); */
    m_sharedData->~NN(); // this frees resources
    RTFree(mWorld, m_sharedData);
  }
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

bool NNUGen::allocBuffers() {
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

void NNUGen::freeBuffers() {
  /* Print("NN: freeing buffers\n"); */
  freeRingBuffer(mWorld, m_inBuffer);
  freeRingBuffer(mWorld, m_outBuffer);
  RTFree(mWorld, m_inModel);
  RTFree(mWorld, m_outModel);
  /* RTFree(mWorld, m_model); */
}

NN::~NN() {
  freeRingBuffer(mWorld, m_inBuffer);
  freeRingBuffer(mWorld, m_outBuffer);
  RTFree(mWorld, m_inModel);
  RTFree(mWorld, m_outModel);
  if(m_compute_thread) { free(m_compute_thread); }
  // thread destroyed by unique_ptr
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

  registerUnit<NN::NNUGen>(ft, "NNUGen", false);
  NN::Cmd::definePlugInCmds();
}

// custom perform method:
// - avoid creating vectors
// - no lock
// - simplified tensor_in reshaping
/* auto const CPU = torch::kCPU; */
/* void SCBackend::perform( */
/*   float* in_buffer, float* out_buffer, int n_vec, */ 
/*   const NNModelMethod* method, int n_batches) const { */

/*   c10::InferenceMode guard; */
/*   int in_dim = method->inDim; */
/*   int in_ratio = method->inRatio; */
/*   int out_dim = method->outDim; */
/*   int out_ratio = method->outRatio; */
/*   auto script_method = m_model.get_method(method->name); */

/*   /1* if (!m_loaded) return; *1/ */

/*   // COPY BUFFER INTO A TENSOR */
/*   /1* auto tensor_in = torch::from_blob(in_buffer, {1, in_dim, n_vec}); *1/ */
/*   /1* tensor_in = tensor_in.reshape({in_dim, n_batches, -1, in_ratio}); *1/ */
/*   auto tensor_in = torch::from_blob(in_buffer, {n_batches, in_dim, n_vec/in_ratio, in_ratio}); */
/*   tensor_in = tensor_in.select(-1, -1); */
/*   /1* tensor_in = tensor_in.permute({1, 0, 2}); *1/ */

/*   // SEND TENSOR TO DEVICE */
/*   /1* std::unique_lock<std::mutex> model_lock(m_model_mutex); *1/ */
/*   tensor_in = tensor_in.to(m_device); */
/*   std::vector<torch::jit::IValue> inputs = {tensor_in}; */

/*   // PROCESS TENSOR */
/*   at::Tensor tensor_out; */
/*   try { */
/*     tensor_out = script_method(inputs).toTensor(); */
/*     tensor_out = tensor_out.repeat_interleave(out_ratio).reshape( */
/*         {n_batches, out_dim, -1}); */
/*   } catch (const std::exception &e) { */
/*     std::cerr << e.what() << '\n'; */
/*     return; */
/*   } */
/*   /1* model_lock.unlock(); *1/ */

/*   int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)), */
/*       out_n_vec(tensor_out.size(2)); */

/*   if (out_n_vec != n_vec) { */
/*     std::cout << "model output size is not consistent, expected " << n_vec */
/*               << " samples, got " << out_n_vec << "!\n"; */
/*     return; */
/*   } */

/*   tensor_out = tensor_out.to(CPU); */
/*   tensor_out = tensor_out.reshape({out_batches * out_channels, -1}); */
/*   auto out_ptr = tensor_out.contiguous().data_ptr<float>(); */

/*   memcpy(out_buffer, out_ptr, n_vec * sizeof(float)); */
/* } */
