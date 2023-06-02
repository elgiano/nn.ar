// NNModel.cpp

#include "NNModel.hpp"
#include "NNUGen.hpp"
#include "NNModelCmd.hpp"
#include "SC_PlugIn.hpp"
#include "backend.h"
#include <cstdio>
#include <fstream>
#include <ostream>
#include <stdexcept>

static InterfaceTable* ft;

namespace NN {


NNModel::NNModel(): m_backend(), m_methods() {
}

NNModel* NNModelRegistry::get(std::string key, bool warn) {
  auto model = models[key];
  if (model == nullptr) {
    if (warn) Print("NNModel: %s not found\n", key.c_str());
    return nullptr;
  }
  if (!model->is_loaded()) {
    if (warn) Print("NNModel: %s not loaded yet\n", key.c_str());
  }
  return model;
}

NNModel* NNModelRegistry::get(unsigned short idx, bool warn) {
  try {
    auto model = modelsByIdx.at(idx);
    if (!model->is_loaded()) {
      if (warn) Print("NNModel: idx %d not loaded yet\n", idx);
    }
    return model;
  }
  catch (const std::out_of_range&) {
    if (warn) Print("NNModel: idx %d not found\n", idx);
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
  Print("NNModel: loading %s\n", path);
  bool loaded = m_backend.load(path) == 0;
  if (loaded) {
    Print("NNModel: loaded %s\n", path);
  } else {
    Print("ERROR: NNModel can't load model %s\n", path);
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

  return true;
}


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
    if (warn) Print("NNModel: method %d not found\n", idx);
    return nullptr;
  }
}

// file dumps are needed to share info with client

void NNModel::streamInfo(std::ostream& stream) {
  stream << "{idx: " << m_idx
    << ", modelPath: '" << m_path << "'"
    << ", minBufferSize: " << m_higherRatio
    << ", methods: [";
  for (auto m: m_methods) {
    stream << "{name: '" << m.name << "'"
      << ", inDim: " << m.inDim
      << ", inRatio: " << m.inRatio
      << ", outDim: " << m.outDim
      << ", outRatio: " << m.outRatio
      << "}, ";
  }
  stream << "]";
  auto settings = m_backend.get_settable_attributes();
  if (settings.size() > 0) {
    stream << ", settings: [";
    for(std::string attr: settings)
      stream << "'" << attr << "', "; 
    stream << "]";
  }
  stream << "}";
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
      Print("ERROR: NNModel couldn't open file %s\n", filename);
      return false;
    }
    streamInfo(file);
    file.close();
    return true;
  }
  catch (const std::exception&) {
    Print("ERROR: NNModel couldn't dump info to file %s\n", filename);
    return false;
  }
}

// UGEN

template <class in_type, class out_type>
bool RTCircularBuffer<in_type, out_type>::initialize(World* world, size_t size) {
  _max_size = size;
  _buffer = (out_type*) RTAlloc(world, sizeof(out_type) * size);
  return _buffer != nullptr;
}

template <class in_type, class out_type>
void RTCircularBuffer<in_type, out_type>::free(World* world) {
  RTFree(world, _buffer);
  _max_size = 0;
}

//
bool NN::loadModel() {
  unsigned short modelIdx = static_cast<unsigned short>(in0(NNInputs::modelIdx));
  unsigned short methodIdx = static_cast<unsigned short>(in0(NNInputs::methodIdx));
  /* Print("NN: getting model n %d\n", modelIdx); */
  m_model = gModels.get(modelIdx);
  if (m_model == nullptr) {
    Print("NN: model %d not found\n", modelIdx);
    return false;
  }
  /* Print("NN: getting method n %d\n", methodIdx); */
  NNModelMethod* method = m_model->getMethod(methodIdx);
  /* Print("NN: got method n %s\n", method->name.c_str()); */
  if (method == nullptr) {
    Print("NN: %d not found\n", methodIdx);
    return false;
  }
  m_method = method->name;
  m_inDim = method->inDim;
  m_outDim = method->outDim;
  return true;
}


NN::NN(): m_compute_thread(nullptr), m_enabled(false) {
  /* Print("NN: Ctor %d\n"); */
  bool success = loadModel();
  if (!success) {
    set_calc_function<NN, &NN::clearOutputs>();
    mDone = true;
    return;
  }
  success = allocBuffers();
  if (!success) {
    freeBuffers();
    Unit* unit = this;
    ClearUnitOnMemFailed;
  }
  /* Print("NN: Ctor done\n"); */
  
  m_enabled = true;
  set_calc_function<NN, &NN::next>();
}

void NN::clearOutputs(int nSamples) {
  ClearUnitOutputs(this, nSamples);
}

bool NN::allocBuffers() {

  m_bufferSize = m_model->m_higherRatio;
  /* Print("NN: alloc buffers: size %d\n", m_bufferSize); */

  m_inBuffer = (RTCircularBuffer<float, float>**) RTAlloc(mWorld, sizeof(RTCircularBuffer<float, float>*) * m_inDim);
  /* Print("NN: inBuffer %p\n", m_inBuffer); */
  if(m_inBuffer == nullptr) return false;
  m_inModel = (float**) RTAlloc(mWorld, sizeof(float) * m_inDim);
  /* Print("NN: inModel %p\n", m_inModel); */
  if(m_inModel == nullptr) return false;
  for (int c(0); c < m_inDim; ++c) {
    m_inBuffer[c] = new RTCircularBuffer<float, float>();
    bool alloc = m_inBuffer[c]->initialize(mWorld, m_bufferSize);
    /* Print("NN: inBuffer %d %d\n", c, alloc); */
    if (!alloc) return false;
    m_inModel[c] = (float *) RTAlloc(mWorld, sizeof(float) * m_bufferSize);
    /* Print("NN: inModel %d %p\n", c, m_inModel[c]); */
    if (m_inModel[c] == nullptr) return false;
  }

  m_outBuffer = (RTCircularBuffer<float, float>**) RTAlloc(mWorld, sizeof(RTCircularBuffer<float, float>*) * m_outDim);
  /* Print("NN: outBuffer %p\n", m_outBuffer); */
  if(m_outBuffer == nullptr) return false;
  m_outModel = (float**) RTAlloc(mWorld, sizeof(float) * m_outDim);
  /* Print("NN: outModel %p\n", m_outModel); */
  if(m_outModel == nullptr) return false;
  for (int c(0); c < m_outDim; ++c) {
    m_outBuffer[c] = new RTCircularBuffer<float, float>();
    bool alloc = m_outBuffer[c]->initialize(mWorld, m_bufferSize);
    /* Print("NN: outBuffer %d %d\n", c, alloc); */
    if (!alloc) return false;
    m_outModel[c] = (float *) RTAlloc(mWorld, sizeof(float) * m_bufferSize);
    /* Print("NN: outModel %d %p\n", c, m_outModel[c]); */
    if(m_outModel[c] == nullptr) return false;
  }
  return true;
}

void NN::freeBuffers() {
  if (m_compute_thread) m_compute_thread->join();
  /* Print("NN: freeing buffers\n"); */
  RTFree(mWorld, m_inModel);
  RTFree(mWorld, m_outModel);

  for (int c(0); c < m_inDim; ++c) {
    m_inBuffer[c]->free(mWorld);
    /* delete m_inModel[c]; */
  }
  RTFree(mWorld, m_inBuffer);
  for (int c(0); c < m_outDim; ++c) {
    m_outBuffer[c]->free(mWorld);
    /* delete m_outModel[c]; */
  }
  RTFree(mWorld, m_outBuffer);
}
NN::~NN() { freeBuffers(); }

void model_perform(NN* nn_instance) {
  std::vector<float*> in_model, out_model;
  for (int c(0); c < nn_instance->m_inDim; c++)
    in_model.push_back(nn_instance->m_inModel[c]);
  for (int c(0); c < nn_instance->m_outDim; c++)
    out_model.push_back(nn_instance->m_outModel[c]);
  nn_instance->m_model->perform(in_model, out_model, nn_instance->m_bufferSize,
                                nn_instance->m_method, 1);
  /* // COPY BUFFER INTO A TENSOR */
  /* int in_dim = nn_instance->m_inDim; */
  /* int out_dim = nn_instance->m_outDim; */
  /* int n_vec = nn_instance->m_bufferSize; */
  /* int n_batches = 1; */
  /* int in_ratio = nn_instance->m_inRatio; */
  /* int out_ratio = nn_instance->m_outRatio; */
  /* std::vector<at::Tensor> tensor_in; */
  /* for (int c(0); c < nn_instance->m_inDim; c++) */
  /*   tensor_in.push_back(torch::from_blob(nn_instance->m_inModel[c], {1, 1, n_vec})); */

  /* auto cat_tensor_in = torch::cat(tensor_in, 1); */
  /* cat_tensor_in = cat_tensor_in.reshape({in_dim, n_batches, -1, in_ratio}); */
  /* cat_tensor_in = cat_tensor_in.select(-1, -1); */
  /* cat_tensor_in = cat_tensor_in.permute({1, 0, 2}); */

  /* /1* if (m_cuda_available) *1/ */
  /* /1*   cat_tensor_in = cat_tensor_in.to(CUDA); *1/ */

  /* std::vector<torch::jit::IValue> inputs = {cat_tensor_in}; */

  /* // PROCESS TENSOR */
  /* at::Tensor tensor_out; */
  /* std::unique_lock<std::mutex> model_lock(m_model_mutex); */
  /* try { */
  /*   tensor_out = m_model.get_method(method)(inputs).toTensor(); */
  /*   tensor_out = tensor_out.repeat_interleave(out_ratio).reshape( */
  /*       {n_batches, out_dim, -1}); */
  /* } catch (const std::exception &e) { */
  /*   std::cerr << e.what() << '\n'; */
  /*   return; */
  /* } */
  /* model_lock.unlock(); */

  /* int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)), */
  /*     out_n_vec(tensor_out.size(2)); */

  /* // CHECKS ON TENSOR SHAPE */
  /* if (out_batches * out_channels != out_buffer.size()) { */
  /*   std::cout << "bad out_buffer size, expected " << out_batches * out_channels */
  /*             << " buffers, got " << out_buffer.size() << "!\n"; */
  /*   return; */
  /* } */

  /* if (out_n_vec != n_vec) { */
  /*   std::cout << "model output size is not consistent, expected " << n_vec */
  /*             << " samples, got " << out_n_vec << "!\n"; */
  /*   return; */
  /* } */

  /* tensor_out = tensor_out.to(CPU); */
  /* tensor_out = tensor_out.reshape({out_batches * out_channels, -1}); */
  /* auto out_ptr = tensor_out.contiguous().data_ptr<float>(); */

  /* for (int i(0); i < out_buffer.size(); i++) { */
  /*   memcpy(out_buffer[i], out_ptr + i * n_vec, n_vec * sizeof(float)); */
  /* } */
}

void NN::next(int nSamples) {
  if (!m_model->is_loaded() || mDone || !m_enabled) {
    ClearUnitOutputs(this, nSamples);
  };

  if (bufferSize() > m_bufferSize) {
    Print("NN: blockSize(%d) larger than model bufferSize(%d), disabling\n", bufferSize(), m_bufferSize);
    m_enabled = false;
  } else {
    m_enabled = true;
  };

  // copy inputs to circular buffer
  for (int c(0); c < m_inDim; ++c) {
    const float* samples = in(NNInputs::inputs + c);
    m_inBuffer[c]->put(samples, bufferSize());
  }

  if (m_inBuffer[0]->full()) {

    if (m_compute_thread) m_compute_thread->join();

    // transfer samples from inBuffer to model inBuf
    for (int c(0); c < m_inDim; ++c)
      m_inBuffer[c]->get(m_inModel[c], m_bufferSize);

    /* Print("NN: performing\n"); */
    model_perform(this);

    // transfer samples from model outBuf to outBuffer
    for (int c(0); c < m_outDim; ++c)
      m_outBuffer[c]->put(m_outModel[c], m_bufferSize);

    m_compute_thread = std::make_unique<std::thread>(model_perform, this);
  }

  // copy circular buf to out
  for (int c(0); c < m_outDim; ++c) {
    float* samples = out(c);
    m_outBuffer[c]->get(samples, bufferSize());
  }
}

// CMD
/* void cleanupLoadMsg(World* world, void* inData) { */
/*   LoadMsgData* data = (LoadMsgData*) inData; */
/*   data->free(world); */
/*   RTFree(world, data); */
/* } */
void onLoadMsg(World* world, void* inUserData, sc_msg_iter* args,
               void* replyAddr) {
    LoadMsgData* data = (LoadMsgData*)RTAlloc(world, sizeof(LoadMsgData));
    data->read(world, args);

    DoAsynchronousCommand(
        world, replyAddr, "nn_load", data,
        doLoadMsg, // stage2 is non real time
        [](World*, void*) {
          return true;
        }, // stage3: RT (completion msg performed if true)
        [](World*, void*) {
          return true;
        }, // stage4: NRT (sends /done if true)
        RTFree, 0, 0);
}

void onQueryMsg(World* world, void*, sc_msg_iter* args, void* replyAddr) {
    const char *key = args->gets();
    if (key) {
      doQueryModel(key);
    } else {
      gModels.printAllInfo();
    }
}

} // namespace NN

PluginLoad(NNUGens) {
  // Plugin magic
  ft = inTable;

  DefinePlugInCmd("/nn_load", NN::onLoadMsg, nullptr);
  DefinePlugInCmd("/nn_query", NN::onQueryMsg, nullptr);
  registerUnit<NN::NN>(ft, "NN", false);
}
