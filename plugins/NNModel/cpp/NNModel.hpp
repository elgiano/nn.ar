// NNModel.hpp

#pragma once
#include "backend.h"
#include <ostream>

namespace NN {

class NNModelMethod {
public:
  NNModelMethod(std::string name, const std::vector<int>& params);

  std::string name;
  int inDim, inRatio, outDim, outRatio;
};

class NNModel {
public:

  NNModel();

  NNModelMethod* getMethod(unsigned short idx, bool warn=true);
  std::string getAttribute(unsigned short idx, bool warn=true) const;
  bool set(std::string attrName, std::string value, bool warn=true);
  bool set(unsigned short attrIdx, std::string value, bool warn=true);
  bool set(unsigned short attrIdx, float value, bool warn=true);
  float get(std::string attrName, bool warn);
  float get(unsigned short attrIdx, bool warn);

  // info
  void streamInfo(std::ostream& dest) const;
  bool dumpInfo(const char* filename) const;
  void printInfo() const;

  // load .ts
  bool load(const char* path);
  bool is_loaded() { return m_backend.is_loaded(); }

  // nn_tilde perform
  void perform(std::vector<float*> inBuffer, std::vector<float*> outBuffer, int n_vec, std::string method, int n_batches) { 
    m_backend.perform(inBuffer, outBuffer, n_vec, method, n_batches);
  }
  // custom perform
  void perform(float* in_buffer, float* out_buffer, int n_vec, const NNModelMethod* method, int n_batches) const;

  void warmup_method(const NNModelMethod* method) const;

  std::string m_path;
  unsigned short m_idx;
  std::vector<NNModelMethod> m_methods;
  std::vector<std::string> m_attributes;
  int m_higherRatio;

  // clone
  NNModel(const NNModel& orig):
    m_path(orig.m_path),
    m_idx(orig.m_idx),
    m_methods(orig.m_methods),
    m_attributes(orig.m_attributes),
    m_higherRatio(orig.m_higherRatio)
  {
    m_backend.m_path = orig.m_backend.m_path;
    m_backend.m_loaded = orig.m_backend.m_loaded;
    m_backend.m_available_methods = orig.m_backend.m_available_methods;
    m_backend.m_device = orig.m_backend.m_device;
    m_backend.m_use_gpu = orig.m_backend.m_use_gpu;
    m_backend.m_model = orig.m_backend.m_model.clone();
  }

private:

  Backend m_backend;
};

class NNModelRegistry {
public:
  NNModelRegistry();
  // load model from .ts file
  NNModel* load(const char* path);
  NNModel* load(unsigned short id, const char* path);
  void unload(unsigned short id);
  // get stored model
  NNModel* get(unsigned short id, bool warn=true) const;
  // all loaded models info
  void streamAllInfo(std::ostream& stream) const;
  bool dumpAllInfo(const char* filename) const;
  void printAllInfo() const;

private:
  unsigned short getNextId();
  std::map<unsigned short, NNModel*> models;
  unsigned short modelCount;

};

} // namespace RAVE
