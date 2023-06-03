#pragma once
#include "NNModel.hpp"
#include "SC_PlugIn.h"

namespace NN {

char* copyStrToBuf(char** buf, const char* str) {
  char* res = strcpy(*buf, str); *buf += strlen(str) + 1;
  return res;
}

// /cmd /nn_set str str str
struct LoadCmdData {
public:
  const char* key;
  const char* path;
  const char* filename;

  static LoadCmdData* nrtalloc(InterfaceTable* ft, sc_msg_iter* args) {

    const char* key = args->gets();
    const char* path = args->gets();
    const char* filename = args->gets();
    if (key == 0 || path == 0) {
      Print("Error: LoadCmd needs a key and a path to a .ts file\n");
      return nullptr;
    }

    size_t dataSize = sizeof(LoadCmdData)
      + strlen(key) + 1
      + strlen(path) + 1
      + strlen(filename) + 1;

    LoadCmdData* cmdData = (LoadCmdData*) NRTAlloc(dataSize);
    if (cmdData == nullptr) {
      Print("LoadCmdData: alloc failed.\n");
      return nullptr;
    }
    char* data = (char*) (cmdData + 1);
    cmdData->key = copyStrToBuf(&data, key);
    cmdData->path = copyStrToBuf(&data, path);
    cmdData->filename = copyStrToBuf(&data, filename);
    return cmdData;
  }

  LoadCmdData() = delete;
};

bool doLoadMsg(World* world, void* inData) {
    LoadCmdData* data = (LoadCmdData*)inData;
    const char* key = data->key;           //.string;
    const char* path = data->path;         //.string;
    const char* filename = data->filename; //.string;

    bool loaded = gModels.load(key, path);

    if (loaded && filename != nullptr) {
      gModels.get(key)->dumpInfo(filename);
    }
    return true;
}

// /cmd /nn_set int int str
struct SetCmdData {
public:
  int modelIdx;
  int settingIdx;
  const char* valueString;

  static SetCmdData* nrtalloc(InterfaceTable* ft, sc_msg_iter* args) {

    int modelIdx = args->geti(-1);
    int settingIdx = args->geti(-1);
    const char* valueString = args->gets();
    if (modelIdx < 0 || settingIdx < 0) {
      Print("Error: SetCmd needs a model and a setting indices\n");
      return nullptr;
    }

    size_t dataSize = sizeof(SetCmdData)
      + strlen(valueString) + 1;

    SetCmdData* cmdData = (SetCmdData*) NRTAlloc(dataSize);
    if (cmdData == nullptr) {
      Print("SetCmdData: alloc failed.\n");
      return nullptr;
    }
    char* data = (char*) (cmdData + 1);
    cmdData->modelIdx = modelIdx;
    cmdData->settingIdx = settingIdx;
    cmdData->valueString = copyStrToBuf(&data, valueString);
    return cmdData;
  }

  SetCmdData() = delete;
};

bool doSetMsg(World* world, void* inData) {
  SetCmdData* data = (SetCmdData*)inData;
  int modelIdx = data->modelIdx;
  int settingIdx = data->settingIdx;
  std::string valueString = data->valueString;

  auto model = gModels.get(modelIdx);
  if (!model) return true;
  model->set(settingIdx, valueString);
  return true;
}

// /cmd /nn_query str
struct QueryCmdData {
public:
  int modelIdx;

  static QueryCmdData* nrtalloc(InterfaceTable* ft, sc_msg_iter* args) {
    int modelIdx = args->geti(-1);
    QueryCmdData* cmdData = (QueryCmdData*) NRTAlloc(sizeof(QueryCmdData));
    if (cmdData == nullptr) {
      Print("QueryCmdData: alloc failed.\n");
      return nullptr;
    }
    cmdData->modelIdx = modelIdx;
    return cmdData;
  }

  QueryCmdData() = delete;
};
bool doQueryMsg(World* world, void* inData) {
  QueryCmdData* data = (QueryCmdData*)inData;
  int modelIdx = data->modelIdx;
  if (modelIdx < 0) {
    gModels.printAllInfo();
    return true;
  }
  const auto model = gModels.get(modelIdx, true);
  if (model)
    model->printInfo();
  return true;
}

}
