#include "NNModel.hpp"
#include <cstdio>
#include <fstream>
#include <ostream>

#include "SC_InterfaceTable.h"
extern "C" InterfaceTable* ft;

namespace NN {

NNModelRegistry::NNModelRegistry(): models(), modelCount(0) {
}

unsigned short NNModelRegistry::getNextId() {
  unsigned short id = modelCount;
  while(models[id] != nullptr) id++;
  return id;
};

NNModel* NNModelRegistry::get(unsigned short id, bool warn) const {
  NNModel* model;
  bool found = false;
  try {
    model = models.at(id);
    found = model != nullptr;
  } catch(...) {
    if (warn) {
      Print("NNBackend: id %d not found. Loaded models:%s\n", id, models.size() ? "" : " []");
      for (auto kv: models) {
        Print("id: %d -> %s\n", kv.first, kv.second->m_path.c_str());
      }
    };
    found = false;
  }

  if (!found) return nullptr;
  if (!model->is_loaded()) {
    if (warn) Print("NNBackend: id %d not loaded yet\n", id);
  }
  return model;
}

void NNModelRegistry::streamAllInfo(std::ostream& dest) const{
  for (const auto kv: models) {
    kv.second->streamInfo(dest);
  }
}

void NNModelRegistry::printAllInfo() const{
  streamAllInfo(std::cout);
  std::cout << std::endl;
}

NNModel* NNModelRegistry::load(const char* path) {
  unsigned short id = getNextId();
  return load(id, path);
}
NNModel* NNModelRegistry::load(unsigned short id, const char* path) {
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

  model = new NNModel();
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

bool NNModelRegistry::dumpAllInfo(const char* filename) const {
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


NNModel::NNModel(): m_backend(), m_methods(), m_path() {}

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

  // cache attributes
  if (m_attributes.size() > 0) m_attributes.clear();
  for (std::string name: m_backend.get_settable_attributes()) {
    m_attributes.push_back(name);
  }

  return true;
}

bool NNModel::set(std::string attrName, std::string value, bool warn) {
  /* Print("setting attr %s to %f\n", name.c_str(), value); */
  std::vector<std::string> args = {value};
  try {
    m_backend.set_attribute(attrName, args);
  } catch (...) {
    if (warn) Print("NNBackend: can't set attribute %s\n", attrName.c_str());
    return false;
  }
  return true;
}
bool NNModel::set(unsigned short attrIdx, std::string value, bool warn) {
  std::string attrName = getAttribute(attrIdx, warn);
  if (attrName.empty()) return false;
  return set(attrName, value, warn);
};
bool NNModel::set(unsigned short attrIdx, float value, bool warn) {
  return set(attrIdx, std::to_string(value), warn);
}

float NNModel::get(std::string name, bool warn) {
  /* Print("attribute %s to %f\n", name.c_str(), value); */
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
float NNModel::get(unsigned short attrIdx, bool warn) {
  std::string attr = getAttribute(attrIdx, warn);
  if (attr.empty()) return false;
  return get(attr, warn);
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

std::string NNModel::getAttribute(unsigned short idx, bool warn) const {
  try {
    return m_attributes.at(idx);
  } catch (const std::out_of_range&) {
    if (warn) Print("NNBackend: attribute %d not found\n", idx);
    return nullptr;
  }
}

// file dumps are needed to share info with client

void NNModel::streamInfo(std::ostream& stream) const {
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

void NNModel::printInfo() const {
  streamInfo(std::cout);
  std::cout << std::endl;
}

bool NNModel::dumpInfo(const char* filename) const {
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
auto const CPU = torch::kCPU;
void NNModel::perform(
  float* in_buffer, float* out_buffer, int n_vec, 
  const NNModelMethod* method, int n_batches) const {

  c10::InferenceMode guard;
  int in_dim = method->inDim;
  int in_ratio = method->inRatio;
  int out_dim = method->outDim;
  int out_ratio = method->outRatio;
  auto script_method = m_backend.m_model.get_method(method->name);
  auto m_device = m_backend.m_device;

  /* if (!m_loaded) return; */

  // COPY BUFFER INTO A TENSOR
  /* auto tensor_in = torch::from_blob(in_buffer, {1, in_dim, n_vec}); */
  /* tensor_in = tensor_in.reshape({in_dim, n_batches, -1, in_ratio}); */
  auto tensor_in = torch::from_blob(in_buffer, {n_batches, in_dim, n_vec/in_ratio, in_ratio});
  tensor_in = tensor_in.select(-1, -1);
  /* tensor_in = tensor_in.permute({1, 0, 2}); */

  // SEND TENSOR TO DEVICE
  // no lock: multiple processes can happen at the same time
  tensor_in = tensor_in.to(m_device);
  std::vector<torch::jit::IValue> inputs = {tensor_in};

  // PROCESS TENSOR
  at::Tensor tensor_out;
  try {
    tensor_out = script_method(inputs).toTensor();
    tensor_out = tensor_out.repeat_interleave(out_ratio).reshape(
        {n_batches, out_dim, -1});
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return;
  }

  int out_batches(tensor_out.size(0)), out_channels(tensor_out.size(1)),
      out_n_vec(tensor_out.size(2));

  if (out_n_vec != n_vec) {
    std::cout << "model output size is not consistent, expected " << n_vec
              << " samples, got " << out_n_vec << "!\n";
    return;
  }

  tensor_out = tensor_out.to(CPU);
  tensor_out = tensor_out.reshape({out_batches * out_channels, -1});
  auto out_ptr = tensor_out.contiguous().data_ptr<float>();

  memcpy(out_buffer, out_ptr, n_vec * sizeof(float));
}

}
