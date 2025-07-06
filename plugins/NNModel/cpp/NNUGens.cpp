// NNUGens.cpp
//
#include "NNModel.hpp"
#include "NNUGens.hpp"
#include "NNModelCmd.hpp"
#include "SC_Unit.h"
#include "SC_InterfaceTable.h"
#include "SC_PlugIn.hpp"
#include "nova-tt/thread_priority.hpp"
#include <chrono>

InterfaceTable* ft;

#define DEBUG(...) if (m_debug >= Debug::all) Print(__VA_ARGS__)

// global model store, by numeric id
NN::NNModelDescLib gModels;
// global debug level
namespace NN { bool gDebug = false; }

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
  int i = UGenInputs::inputs + m_inDim * m_batches;
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
    if (!attr.changed()) continue;
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

void model_perform_load(NN* nn, int warmup) {
  auto path = nn->m_modelDesc->getPath();
  auto m_debug = nn->m_debug;
  DEBUG("NNUGen: loading model %s\n", path);
  int err = nn->m_model.load(path);
  if (err) {
    Print("NNUGen: ERROR loading model %s\n", path);
    return;
  }
  if (warmup > 0) {
    DEBUG("NNUGen: warming up model\n", path);
    nn->warmupModel(warmup);
  }
  nn->m_loaded = true;
  DEBUG("NNUGen: loaded %s\n", path);
}

void model_perform_cleanup(NN* nn_instance) {
  auto mWorld = nn_instance->mWorld;
  // manually call destructor and free instance
  nn_instance->~NN();
  RTFree(mWorld, nn_instance);
}

void model_perform(NN* nn_instance) {
  /* Timer timer; */
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim * nn_instance->m_batches; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < nn_instance->m_outDim * nn_instance->m_batches; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
  model_perform_attributes(nn_instance);
  /* timer.print("attrs:"); */
  nn_instance->m_model.perform(in_model, out_model,
                               nn_instance->m_bufferSize,
                               nn_instance->m_method->name, nn_instance->m_batches);
  /* timer.print("perform:"); */
}

inline void set_thread_rt_prio(const NN* nn_instance) {
#ifdef NOVA_TT_PRIORITY_RT
    int priority = nova::thread_priority_interval_rt().first;
    auto m_debug = nn_instance->m_debug;
    DEBUG("NNUGen: setting thread rt prio (%d)\n", priority);
    nova::thread_set_priority_rt(priority);
#elif defined(NOVA_TT_PRIORITY_PERIOD_COMPUTATION_CONSTRAINT)
    double ns_per_block = nn_instance->m_nsPerBlock; 
    int success;
    #    ifdef __APPLE__
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double ns_to_host = static_cast<double>(timebase.denom) / timebase.numer;
    int computation = ns_to_host * (ns_per_block - 5000);
    int constraint = ns_to_host * ns_per_block;
    auto m_debug = nn_instance->m_debug;
    DEBUG("NNUGen: setting thread rt prio (ns_to_host: %f; computation: %d, constraint: %d)\n",
          ns_to_host, computation, constraint);
    success = nova::thread_set_priority_rt(0, computation, constraint, true);
#    else
    success = nova::thread_set_priority_rt(ns_per_block, ns_per_block - 2, ns_per_block - 1, false);
#    endif
    if (!success)
        Print("WARNING: NNUGen: couldn't set thread rt prio\n");
#endif
}

void model_perform_loop(NN *nn_instance, int warmup) {

  set_thread_rt_prio(nn_instance);
  model_perform_load(nn_instance, warmup);
  std::vector<float *> in_model, out_model;
  int numInputs = nn_instance->m_inDim * nn_instance->m_batches;
  int numOutputs = nn_instance->m_outDim * nn_instance->m_batches;
  for (int c(0); c < numInputs; ++c)
    in_model.push_back(&nn_instance->m_inModel[nn_instance->m_bufferSize * c]);
  for (int c(0); c < numOutputs; ++c)
    out_model.push_back(&nn_instance->m_outModel[nn_instance->m_bufferSize * c]);
  while (!nn_instance->m_should_stop_perform_thread) {
    if (nn_instance->m_data_available_lock.try_acquire_for(
      std::chrono::milliseconds(200))) {
        /* nn_instance->timer.print("received in:"); */
      model_perform_attributes(nn_instance);
        /* Timer timer; */
      nn_instance->m_model.perform(in_model, out_model,
                                   nn_instance->m_bufferSize,
                                   nn_instance->m_method->name, nn_instance->m_batches);
        /* timer.print("model perform:"); */
      nn_instance->m_result_available_lock.release();
    }
  }
  model_perform_cleanup(nn_instance);
  auto m_debug = nn_instance->m_debug;
  DEBUG("NN: thread exit\n");
}

void NNUGen::next(int nSamples) {

  if (!m_sharedData->m_loaded) {
    ClearUnitOutputs(this, nSamples);
    return;
  };

  // update attr setters
  for (auto& a: m_sharedData->m_attributes) a.update(this, nSamples);

  int numInputs = m_inDim * m_batches;
  int numOutputs = m_outDim * m_batches;

  // DEBUG("NNUGen next: writing %d\n", nSamples);

  // copy inputs to circular buffer
  if(isControlRateIn(UGenInputs::inputs)) {
    for (int c(0); c < numInputs; ++c)
      m_inBuffer[c].putRepeat(in0(UGenInputs::inputs + c), nSamples);
  } else {
    for (int c(0); c < numInputs; ++c)
      m_inBuffer[c].put(in(UGenInputs::inputs + c), nSamples);
  }

  if (m_inBuffer[0].full()) {
    // DEBUG("NNUGen inBuf full: sending to model\n");

    if (!m_useThread) {

      for (int c(0); c < numInputs; ++c)
        m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);

      model_perform(m_sharedData);

      for (int c(0); c < numOutputs; ++c)
        m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);
    } else if (m_sharedData->m_result_available_lock.try_acquire()) {
      /* Print("sending\n"); m_sharedData->timer.reset(); */
      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < numInputs; ++c)
        m_inBuffer[c].get(&m_inModel[c * m_bufferSize], m_bufferSize);
      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      
      // DEBUG("NNUGen reading model outputs\n");
      for (int c(0); c < numOutputs; ++c)
        m_outBuffer[c].put(&m_outModel[c * m_bufferSize], m_bufferSize);
      // SIGNAL PERFORM THREAD THAT DATA IS AVAILABLE
      m_sharedData->m_data_available_lock.release();
    }
  }

  // DEBUG("NNUGen inBuf size: %d\nNNUGen outBuf size: %d\nNNUGen outBuf reading %d\n",
        // m_inBuffer->readable(), m_outBuffer->readable(), nSamples);
  // copy circular buf to out
  for (int c(0); c < numOutputs; ++c)
    m_outBuffer[c].get(out(c), nSamples);
}

NN::NN(
  World* world,
  const NNModelDesc* modelDesc, const NNModelMethod* modelMethod,
  float* inModel, float* outModel,  
  RingBuf* inRing, RingBuf* outRing,
  int bufferSize, int debug, int batches, int nsPerBlock): 
  mWorld(world),
  m_inModel(inModel), m_outModel(outModel),
  m_inBuffer(inRing), m_outBuffer(outRing),
  m_method(modelMethod), m_modelDesc(modelDesc), 
  m_bufferSize(bufferSize), m_debug(debug),
  m_batches(batches),
  m_compute_thread(nullptr),
  m_data_available_lock(0), m_result_available_lock(1),
  m_should_stop_perform_thread(false), m_loaded(false),
  m_nsPerBlock(nsPerBlock)
{
  m_inDim = m_method->inDim;
  m_outDim = m_method->outDim;
}


NNUGen::NNUGen(): 
  m_inBuffer(nullptr), m_outBuffer(nullptr), m_sharedData(nullptr)
{
  m_debug = static_cast<int>(in0(UGenInputs::debug));
  auto modelIdx = static_cast<unsigned short>(in0(UGenInputs::modelIdx));
  const NNModelDesc* modelDesc = gModels.get(modelIdx);
  const NNModelMethod* modelMethod = nullptr;
  if (modelDesc)
    modelMethod = getModelMethod(modelDesc, in0(UGenInputs::methodIdx));
  if (modelMethod == nullptr) {
    Print("ERROR: (scsynth) NNUGen no model method, clearing\n");
    set_calc_function<NNUGen, &NNUGen::clearOutputs>();
    return;
  }
  m_inDim = modelMethod->inDim;
  m_outDim = modelMethod->outDim;
  m_batches = sc_max(1, static_cast<int>(in0(UGenInputs::n_batches)));

  m_bufferSize = in0(UGenInputs::bufSize);
  DEBUG("NNUGen: bufSize %d\n", m_bufferSize); 

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

  void* data = RTAlloc(mWorld, sizeof(NN));
  if (!data) {
    freeBuffers();
    ClearUnitOnMemFailed;
  }
  DEBUG("NNUGen: init sharedData\n");
  int nsPerBlock = 1e9 * fullSampleDur() * fullBufferSize();
  m_sharedData = new(data) NN(mWorld, modelDesc, modelMethod, 
                        m_inModel, m_outModel, m_inBuffer, m_outBuffer,
                        m_bufferSize, m_debug, m_batches, nsPerBlock);

  DEBUG("NNUGen: use thread %d\n", m_useThread);
  int warmup = static_cast<int>(in0(UGenInputs::warmup));
  if (m_useThread)
    m_sharedData->m_compute_thread = new std::thread(model_perform_loop, m_sharedData, warmup);
  else
    model_perform_load(m_sharedData, warmup);

  DEBUG("NNUGen: setupAttributes\n", m_useThread);
  setupAttributes();

  mCalcFunc = make_calc_function<NNUGen, &NNUGen::next>();
  DEBUG("NNUGen: Ctor done\n");
}

NNUGen::~NNUGen() {
  DEBUG("NNUGen: Dtor\n");
  if (m_sharedData && m_sharedData->m_compute_thread) {
    // don't wait for join, it would stall the dsp chain
    // thread frees resources when stopped
    m_sharedData->m_should_stop_perform_thread = true;
    /* m_compute_thread->join(); */
  } else {
    DEBUG("NNUGen: freeing manually\n");
    if (m_sharedData) m_sharedData->~NN(); // this frees resources
    RTFree(mWorld, m_sharedData);
  }
}

// BUFFERS

RingBuf* allocRingBuffer(World* world, size_t bufSize, size_t numChannels) {
  size_t memSize = (sizeof(RingBuf) + sizeof(float) * bufSize) * numChannels;
  RingBuf* ctrs = (RingBuf*) RTAlloc(world, memSize);
  if (ctrs == nullptr) return nullptr;
  float* data = reinterpret_cast<float*>(ctrs + numChannels);
  memset(data, 0, sizeof(float) * numChannels * bufSize);
  for (int c(0); c < numChannels; ++c) {
    auto buf = data + (c * bufSize);
    new(ctrs + c) RingBuf(buf, bufSize);
  }
  return ctrs;
}

void freeRingBuffer(World* world, RingBuf* buf) {
  RTFree(world, buf);
}

bool NNUGen::allocBuffers() {
  int numInputs = m_inDim * m_batches;
  int numOutputs = m_outDim * m_batches;
  m_inBuffer = allocRingBuffer(mWorld, m_bufferSize, numInputs);
  if (m_inBuffer == nullptr) return false;
  m_outBuffer = allocRingBuffer(mWorld, m_bufferSize, numOutputs);
  if (m_outBuffer == nullptr) return false;
  m_inModel = rtAlloc<float>(mWorld, m_bufferSize * numInputs);
  if(m_inModel == nullptr) return false;
  m_outModel = rtAlloc<float>(mWorld, m_bufferSize * numOutputs);
  if(m_outModel == nullptr) return false;
  memset(m_inModel, 0, sizeof(float) * m_bufferSize * numInputs);
  memset(m_outModel, 0, sizeof(float) * m_bufferSize * numOutputs);
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
  if (m_compute_thread) { free(m_compute_thread); }
}

void NN::warmupModel(int n_passes=1) {
  /* Timer timer; */
  std::vector<float *> in_model, out_model;
  for (int c(0); c < m_inDim; ++c)
    in_model.push_back(&m_inModel[m_bufferSize * c]);
  for (int c(0); c < m_outDim; ++c)
    out_model.push_back(&m_outModel[m_bufferSize * c]);

  for(int i=0; i < n_passes; ++i)
    m_model.perform(in_model, out_model, m_bufferSize, m_method->name, 1);
  /* timer.print("warmup:"); */
}

} // namespace NN


PluginLoad(NNUGens) {
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
