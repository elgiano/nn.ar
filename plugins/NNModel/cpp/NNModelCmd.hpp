#pragma once
#include "NNModel.hpp"
#include "SC_PlugIn.h"

namespace NN {

/* struct CmdString { */
/*   static InterfaceTable* ft; */
/*   char* string; */
/*   void alloc(World *world, const char* s) { */
/*     string = (char *) RTAlloc(world, strlen(s) + 1); */
/*     strcpy(string, s); */
/*   }; */
/*   void free(World *world){ RTFree(world, string); }; */
/* }; */

/* class AsyncCmd { */
/* public: */  
/*   static InterfaceTable* ft; */
/*   AsyncCmd(); */
/*   ~AsyncCmd(); */

/*   bool checkInputs(void* data); */
/*   bool Stage2(World* world, void* data); */
/*   bool Stage3(World* world, void* data); */
/*   bool Stage4(World* world, void* data); */
/*   void cleanup(World* world, void* data); */

/*   const char* allocString(World* world, const char* s) { */
/*     char* buf = (char*) RTAlloc(world, strlen(s) + 1); */
/*     if (buf != nullptr) strcpy(buf, s); */
/*     return (const char*) buf; */
/*   } */
/* private: */

/*   void* data; */
/* }; */

/* class LoadCmd : public AsyncCmd { */
/*   struct Data { */
/*     char* key; */
/*     char* path; */
/*     char* filename; */
/*   }; */ 
  
/*   LoadCmd(World* world, void *inData, sc_msg_iter* args) { */
/*     const char* key = args->gets(); */ 
/*     const char* path = args->gets(); */ 
/*     const char* filename = args->gets(); */ 
/*     if (key == 0 || path == 0) { */
/*       Print("Error: LoadCmd needs a key and a path to a .ts file\n"); */
/*       return; */
/*     } */
/*     size_t memSize = strlen(key) + strlen(path) + strlen(filename) + 3; */
/*     void* memData = RTAcll */
/*   }; */

/*   cleanup(World* world, void *inData) { */

/*   } */

/*   Stage2(World* world, void *inData) { */
/*     LoadMsgData* data = (LoadMsgData*)inData; */
/*     const char* key = data->key;           //.string; */
/*     const char* path = data->path;         //.string; */
/*     const char* filename = data->filename; //.string; */

/*     bool loaded = gModels.load(key, path); */

/*     if (loaded && filename != nullptr) { */
/*       gModels.get(key)->dumpInfo(filename); */
/*     } */
/*     return true; */
/*   } */
/* private: */
/*   Data* data; */
/* }; */

struct LoadMsgData {
  /* MsgString key; */
  /* MsgString path; */
  /* MsgString filename; */
  const char* key;
  const char* path;
  const char* filename;

  void read(World *world, sc_msg_iter* args) {
    /* key.alloc(world, args->gets()); */
    /* path.alloc(world, args->gets()); */
    /* filename.alloc(world, args->gets()); */
    key = args->gets();
    path = args->gets();
    filename = args->gets();
  };

  /* void free(World *world) { */
  /*   key.free(world); */
  /*   path.free(world); */
  /*   filename.free(world); */
  /* }; */

};


char* copyStrToBuf(char** buf, const char* str) {
  char* res = strcpy(*buf, str); *buf += strlen(str) + 1;
  return res;
}

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

struct SetMsgData {
  int modelIdx;
  unsigned short setting;
  float value;

  bool read(World *world, sc_msg_iter* args) {
    modelIdx = args->geti(-1);
    setting = args->geti(-1);
    value = args->getf();
    if (modelIdx < 0 || setting < 0) {
      return false;
    }
    return true;
  };
};

bool doSetMsg(World* world, void* inData) {
  SetMsgData* data = (SetMsgData*)inData;
  int modelIdx = data->modelIdx;
  int settingIdx = data->setting;
  float value = data->value;

  auto model = gModels.get(modelIdx);
  if (!model) return true;
  model->set(settingIdx, value);
  return true;
}


void doQueryModel(const char* key) {
    const auto model = gModels.get(key, true);
    if (!model)
      return;
    model->printInfo();
}


}
