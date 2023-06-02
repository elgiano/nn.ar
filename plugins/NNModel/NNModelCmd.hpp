#pragma once
#include "NNModel.hpp"
#include "SC_PlugIn.h"

namespace NN {

/* struct MsgString { */
/*   char* string; */
/*   void alloc(World *world, const char* s) { */
/*     string = (char *) RTAlloc(world, strlen(s) + 1); */
/*     strcpy(string, s); */
/*   }; */
/*   void free(World *world){ RTFree(world, string); }; */
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


bool doLoadMsg(World* world, void* inData) {
    LoadMsgData* data = (LoadMsgData*)inData;
    const char* key = data->key;           //.string;
    const char* path = data->path;         //.string;
    const char* filename = data->filename; //.string;

    bool loaded = gModels.load(key, path);

    if (loaded && filename != nullptr) {
      gModels.get(key)->dumpInfo(filename);
    }
    return true;
}


void doQueryModel(const char* key) {
    const auto model = gModels.get(key, true);
    if (!model)
      return;
    model->printInfo();
}

}
