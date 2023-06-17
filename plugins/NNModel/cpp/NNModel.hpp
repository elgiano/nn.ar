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
  ~NNModel();

  NNModelMethod* getMethod(unsigned short idx, bool warn=true);
  std::string getAttribute(unsigned short idx, bool warn=true);
  bool set(std::string attrName, std::string value, bool warn=true);
  bool set(unsigned short attrIdx, std::string value, bool warn=true);
  bool set(unsigned short attrIdx, float value, bool warn=true);
  float get(std::string attrName, bool warn);
  float get(unsigned short attrIdx, bool warn);

  // info
  void streamInfo(std::ostream& dest);
  bool dumpInfo(const char* filename);
  void printInfo();

  // load .ts
  bool load(const char* path);
  bool is_loaded() { return m_backend.is_loaded(); }
  void perform(std::vector<float*> inBuffer, std::vector<float*> outBuffer, int n_vec, std::string method, int n_batches) { 
    m_backend.perform(inBuffer, outBuffer, n_vec, method, n_batches);
  }

  std::string m_path;
  unsigned short m_idx;
  std::vector<NNModelMethod> m_methods;
  std::vector<std::string> m_attributes;
  int m_higherRatio;

private:

  Backend m_backend;
};

class NNModelRegistry {
public:
  NNModelRegistry();
  // load model from .ts file
  NNModel* load(const char* path);
  NNModel* load(unsigned short id, const char* path);
  // get stored model
  NNModel* get(unsigned short id, bool warn=true);
  // all loaded models info
  void streamAllInfo(std::ostream& stream);
  bool dumpAllInfo(const char* filename);
  void printAllInfo();

private:
  unsigned short getNextId();
  std::map<unsigned short, NNModel*> models;
  unsigned short modelCount;

};

} // namespace RAVE
