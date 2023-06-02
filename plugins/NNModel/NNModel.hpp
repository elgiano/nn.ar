// NNModel.h

#pragma once
#include "backend.h"
#include "SC_PlugIn.h"
#include <ostream>

namespace NN {

class NNModelMethod {
public:
  NNModelMethod(std::string name, const std::vector<int>& params);

  std::string name;
  int inDim, inRatio, outDim, outRatio;
};

class NNModel;

class NNModelRegistry {
public:
  NNModelRegistry(): models(), modelsByIdx(), modelCount(0) {}
  // load model from .ts file
  bool load(std::string key, const char* path);
  // get stored model
  NNModel* get(std::string key, bool warn=true);
  NNModel* get(unsigned short idx, bool warn=true);
  void streamAllInfo(std::ostream& stream);
  void printAllInfo();

private:
  std::map<std::string, NNModel*> models;
  std::vector<NNModel*> modelsByIdx;
  unsigned short modelCount;

};
// global model store, by key and index
static NNModelRegistry gModels;

class NNModel {
public:

  NNModel();
  ~NNModel();

  NNModelMethod* getMethod(unsigned short idx, bool warn=true);

  void streamInfo(std::ostream& dest);
  bool dumpInfo(const char* filename);
  void printInfo();

  // load .ts
  bool load(const char* path);
  bool is_loaded() { return m_backend.is_loaded(); }
  void perform(std::vector<float*> inBuffer, std::vector<float*> outBuffer, int n_vec, std::string method, int n_batches) { 
    m_backend.perform(inBuffer, outBuffer, n_vec, method, n_batches);
  }

  const char* m_path;
  unsigned short m_idx;
  std::vector<NNModelMethod> m_methods;
  int m_higherRatio;

private:

  Backend m_backend;
};

} // namespace RAVE
