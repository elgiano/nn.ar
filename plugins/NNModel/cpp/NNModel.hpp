// NNModel.hpp

#pragma once
#include <ostream>
#include <vector>
#include <map>

namespace NN {

// store info about a model method
class NNModelMethod {
public:
  // read method params from model
  NNModelMethod(const std::string& name, const std::vector<int>& params);

  std::string name;
  int inDim, inRatio, outDim, outRatio;
};

// read and store model information
class NNModelDesc {
public:

  // load .ts, just to read info
  bool load(const char* path);
  
  // info
  bool is_loaded() { return m_loaded; }
  void streamInfo(std::ostream& dest) const;
  bool dumpInfo(const char* filename) const;
  void printInfo() const;

  NNModelMethod* getMethod(unsigned short idx, bool warn=true);
  std::string getAttributeName(unsigned short idx, bool warn=true) const;

  std::string m_path;
  unsigned short m_idx;
  std::vector<NNModelMethod> m_methods;
  std::vector<std::string> m_attributes;
  int m_higherRatio;
  bool m_loaded = false;
};

// register model info by int id
class NNModelDescLib {
public:
  NNModelDescLib();
  // load model from .ts file
  NNModelDesc* load(const char* path);
  NNModelDesc* load(unsigned short id, const char* path);
  void unload(unsigned short id);
  /* void reload(unsigned short id); */

  // get stored model
  NNModelDesc* get(unsigned short id, bool warn=true) const;
  // all loaded models info
  void streamAllInfo(std::ostream& stream) const;
  bool dumpAllInfo(const char* filename) const;
  void printAllInfo() const;

private:
  unsigned short getNextId();
  std::map<unsigned short, NNModelDesc*> models;
  unsigned short modelCount;

};

} // namespace RAVE
