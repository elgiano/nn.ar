#include "NNModel.hpp"
#include <cstdio>
#include <fstream>
#include <ostream>
#include "SC_InterfaceTable.h"

extern InterfaceTable* ft;

namespace NN {

NNModelMethod::NNModelMethod(std::string name, const std::vector<int>& params):
name(name) {
  inDim = params[0];
  inRatio = params[1];
  outDim = params[2];
  outRatio = params[3];
}

bool NNModelDesc::load(const char* path) {
  Print("NNModelDesc: loading %s\n", path);
  Backend backend;
  bool loaded = backend.load(path) == 0;
  if (loaded) {
    Print("NNModelDesc: loaded %s\n", path);
  } else {
    Print("ERROR: NNModelDesc can't load model %s\n", path);
    return false;
  }

  // cache path
  m_path = path;

  m_higherRatio = backend.get_higher_ratio();

  // cache methods
  if (m_methods.size() > 0) m_methods.clear();
  for (std::string name: backend.get_available_methods()) {
    auto params = backend.get_method_params(name);
    // skip methods with no params
    if (params.size() == 0) continue;
    NNModelMethod m(name, params);
    m_methods.push_back(m);
  }

  // cache attributes
  if (m_attributes.size() > 0) m_attributes.clear();
  for (std::string name: backend.get_settable_attributes()) {
    m_attributes.push_back(name);
  }

  m_loaded = true;
  return true;
}

NNModelMethod* NNModelDesc::getMethod(unsigned short idx, bool warn) {
  try {
    return &m_methods.at(idx);
  } catch (const std::out_of_range&) {
    if (warn) Print("NNModelDesc: method %d not found\n", idx);
    return nullptr;
  }
}

std::string NNModelDesc::getAttributeName(unsigned short idx, bool warn) const {
  try {
    return m_attributes.at(idx);
  } catch (const std::out_of_range&) {
    if (warn) Print("NNBackend: attribute %d not found\n", idx);
    return "";
  }
}


NNModelDescLib::NNModelDescLib(): models(), modelCount(0) {
}

unsigned short NNModelDescLib::getNextId() {
  unsigned short id = modelCount;
  while(models[id] != nullptr) id++;
  return id;
};

NNModelDesc* NNModelDescLib::get(unsigned short id, bool warn) const {
  NNModelDesc* model;
  bool found = false;
  try {
    model = models.at(id);
    found = model != nullptr;
  } catch(...) {
    if (warn) {
      Print("NNModelDescLib: id %d not found. Loaded models:%s\n", id, models.size() ? "" : " []");
      for (auto kv: models) {
        Print("id: %d -> %s\n", kv.first, kv.second->m_path.c_str());
      }
    };
    found = false;
  }

  if (!found) return nullptr;
  if (!model->is_loaded()) {
    if (warn) Print("NNModelDescLib: id %d not loaded yet\n", id);
  }
  return model;
}

void NNModelDescLib::streamAllInfo(std::ostream& dest) const{
  for (const auto kv: models) {
    kv.second->streamInfo(dest);
  }
}

void NNModelDescLib::printAllInfo() const{
  streamAllInfo(std::cout);
  std::cout << std::endl;
}

NNModelDesc* NNModelDescLib::load(const char* path) {
  unsigned short id = getNextId();
  return load(id, path);
}
NNModelDesc* NNModelDescLib::load(unsigned short id, const char* path) {
  auto model = get(id, false);
  /* Print("NNBackend: loading model %s at idx %d\n", path, id); */
  if (model != nullptr) {
    if (model->m_path == path) {
      Print("NNBackend: model %d already loaded %s\n", id, path);
      return model;
    } else {
      return model->load(path) ? model : nullptr;
    }
  }

  model = new NNModelDesc();
  if (model->load(path)) {
    models[id] = model;
    model->m_idx = id;
    modelCount++;
    return model;
  } else {
    /* delete model; */
    return nullptr;
  }
}

void NNModelDescLib::unload(unsigned short id) {
  auto model = get(id, true);
  /* Print("NNBackend: loading model %s at idx %d\n", path, id); */
  if (model == nullptr) return;

  models.erase(id);
  delete model;
}

bool NNModelDescLib::dumpAllInfo(const char* filename) const {
  std::ofstream file;
  try {
    file.open(filename);
    if (!file.is_open()) {
      Print("ERROR: NNBackend couldn't open file %s\n", filename);
      return false;
    }
    streamAllInfo(file);
    file.close();
    return true;
  }
  catch (...) {
    Print("ERROR: NNBackend couldn't dump info to file %s\n", filename);
    return false;
  }
}

// file dumps are needed to share info with client

void NNModelDesc::streamInfo(std::ostream& stream) const {
  stream << "- idx: " << m_idx
    << "\n  modelPath: " << m_path.c_str()
    << "\n  minBufferSize: " << m_higherRatio
    << "\n  methods:";
  for (auto m: m_methods) {
    stream << "\n    - name: " << m.name
      << "\n      inDim: " << m.inDim
      << "\n      inRatio: " << m.inRatio
      << "\n      outDim: " << m.outDim
      << "\n      outRatio: " << m.outRatio;
  }
  if (m_attributes.size() > 0) {
    stream << "\n  attributes:";
    for(std::string attr: m_attributes)
      stream << "\n    - " << attr; 
  }
  stream << "\n";
}

void NNModelDesc::printInfo() const {
  streamInfo(std::cout);
  std::cout << std::endl;
}

bool NNModelDesc::dumpInfo(const char* filename) const {
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

// custom perform method:
// - avoid creating vectors
// - no lock
// - simplified tensor_in reshaping
/* auto const CPU = torch::kCPU; */
/* void NNModel::perform( */
/*   float* in_buffer, float* out_buffer, int n_vec, */ 
/*   const NNModelMethod* method, int n_batches) const { */

/*   c10::InferenceMode guard; */
/*   int in_dim = method->inDim; */
/*   int in_ratio = method->inRatio; */
/*   int out_dim = method->outDim; */
/*   int out_ratio = method->outRatio; */
/*   auto script_method = m_backend.m_model.get_method(method->name); */
/*   auto m_device = m_backend.m_device; */

/*   /1* if (!m_loaded) return; *1/ */

/*   // COPY BUFFER INTO A TENSOR */
/*   /1* auto tensor_in = torch::from_blob(in_buffer, {1, in_dim, n_vec}); *1/ */
/*   /1* tensor_in = tensor_in.reshape({in_dim, n_batches, -1, in_ratio}); *1/ */
/*   auto tensor_in = torch::from_blob(in_buffer, {n_batches, in_dim, n_vec/in_ratio, in_ratio}); */
/*   tensor_in = tensor_in.select(-1, -1); */
/*   /1* tensor_in = tensor_in.permute({1, 0, 2}); *1/ */

/*   // SEND TENSOR TO DEVICE */
/*   // no lock: multiple processes can happen at the same time */
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

/* void NNModel::warmup_method(const NNModelMethod* method) const { */

/*   Print("NNBackend: warming up method '%s'\n", method->name.c_str()); */
/*   c10::InferenceMode guard; */
/*   int in_dim = method->inDim; */
/*   int in_size = m_higherRatio / method->inRatio; */
/*   auto script_method = m_backend.m_model.get_method(method->name); */
/*   auto m_device = m_backend.m_device; */

/*   /1* if (!m_loaded) return; *1/ */

/*   auto tensor_in = torch::zeros({1, in_dim, in_size}); */
/*     /1* std::cout << "hr: " << m_higherRatio << " ratio: " << method->inRatio << std::endl; *1/ */
/*     /1* std::cout << "in: " << tensor_in.sizes() << std::endl; *1/ */
/*   // SEND TENSOR TO DEVICE */
/*   // no lock: multiple processes can happen at the same time */
/*   tensor_in = tensor_in.to(m_device); */
/*   std::vector<torch::jit::IValue> inputs = {tensor_in}; */

/*   // PROCESS TENSOR */
/*   try { */
/*     script_method(inputs); */
/*     /1* std::cout << "out: " << tensor_out.sizes() << std::endl; *1/ */
/*   } catch (const std::exception &e) { */
/*     std::cerr << e.what() << '\n'; */
/*     return; */
/*   } */
/* } */

}
