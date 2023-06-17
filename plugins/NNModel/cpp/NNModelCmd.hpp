// NNModelCmd.hpp

#pragma once
#include "NNModel.hpp"
#include "SC_PlugIn.h"

namespace NN {

inline char* copyStrToBuf(char** buf, const char* str) {
  char* res = strcpy(*buf, str); *buf += strlen(str) + 1;
  return res;
}

// /cmd /nn_set str str str
struct LoadCmdData {
public:
  int id;
  const char* path;
  const char* filename;

  static LoadCmdData* alloc(sc_msg_iter* args, InterfaceTable* ft, World* world=nullptr) {

    int id = args->geti(-1);
    const char* path = args->gets();
    const char* filename = args->gets("");

    if (path == 0) {
      Print("Error: nn_load needs a path to a .ts file\n");
      return nullptr;
    }

    size_t dataSize = sizeof(LoadCmdData)
      + strlen(path) + 1
      + strlen(filename) + 1;

    LoadCmdData* cmdData = (LoadCmdData*) (world ? RTAlloc(world, dataSize) : NRTAlloc(dataSize));
    if (cmdData == nullptr) {
      Print("nn_load: msg data alloc failed.\n");
      return nullptr;
    }

    char* data = (char*) (cmdData + 1);
    cmdData->id = id;
    cmdData->path = copyStrToBuf(&data, path);
    cmdData->filename = copyStrToBuf(&data, filename);
    return cmdData;
  }

  LoadCmdData() = delete;
};


// /cmd /nn_set int int str
struct SetCmdData {
public:
  int modelIdx;
  int attrIdx;
  const char* valueString;

  static SetCmdData* alloc(sc_msg_iter* args, InterfaceTable* ft, World* world=nullptr) {

    int modelIdx = args->geti(-1);
    int attrIdx = args->geti(-1);
    const char* valueString = args->gets();
    if (modelIdx < 0 || attrIdx < 0) {
      Print("Error: SetCmd needs a model and attribute indices\n");
      return nullptr;
    }

    size_t dataSize = sizeof(SetCmdData)
      + strlen(valueString) + 1;

    SetCmdData* cmdData = (SetCmdData*) (world ? RTAlloc(world, dataSize) : NRTAlloc(dataSize));
    if (cmdData == nullptr) {
      Print("SetCmdData: alloc failed.\n");
      return nullptr;
    }
    char* data = (char*) (cmdData + 1);
    cmdData->modelIdx = modelIdx;
    cmdData->attrIdx = attrIdx;
    cmdData->valueString = copyStrToBuf(&data, valueString);
    return cmdData;
  }

  SetCmdData() = delete;
};


// /cmd /nn_query str
struct QueryCmdData {
public:
  int modelIdx;
  const char* outFile;

  static QueryCmdData* alloc(sc_msg_iter* args, InterfaceTable* ft, World* world=nullptr) {
    int modelIdx = args->geti(-1);
    const char* outFile = args->gets("");

    auto dataSize = sizeof(QueryCmdData) + strlen(outFile) + 1;
    QueryCmdData* cmdData = (QueryCmdData*) (world ? RTAlloc(world, dataSize) : NRTAlloc(dataSize));
    if (cmdData == nullptr) {
      Print("QueryCmdData: alloc failed.\n");
      return nullptr;
    }
    cmdData->modelIdx = modelIdx;
    char* data = (char*) (cmdData + 1);
    cmdData->outFile = copyStrToBuf(&data, outFile);
    
    return cmdData;
  }

  QueryCmdData() = delete;
};

}
