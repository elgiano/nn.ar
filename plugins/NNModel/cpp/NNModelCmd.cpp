#include "NNModelCmd.hpp"
#include "NNModel.hpp"
#include "SC_InterfaceTable.h"
#include "sc_msg_iter.h"
#include <torch/version.h>

extern InterfaceTable* ft;
extern NN::NNModelDescLib gModels;

namespace NN::Cmd {

// we need async commands in order not to block audio thread when loading/querying/unloading models
// otherwise current sound processing crackles while doing it

// defining async commands is complicated:
// the common interface requires to use DefinePlugInCmd and DoAsynchronousCommand from ftTable
// - we need to pass osc args as void* inData
// - we need to copy inData, as it's not guaranteed to survive past Stage1
// - if we want to SendReply, we need to copy and pass replyAddress to the actual stage function
// - we can't actually use SendReply, it's private
// our strategy:
// - accept we can't send data back to sclang:
// e.g. we write a yaml file, sclang decides the path, so it can check it after
// - we pass around sc_msg_iter data, storing as { size_t size, const char* data }
// - msg_iter is re-constructed at each step from size and data
// - todo: completion message? we don't need it for now
// - todo: sendReply? we can't: not there now, and later it would break retrocompat

template<class Cmd>
struct BaseAsyncCmd {
private:
  size_t oscDataSize;
  const char* oscData;
public:

  void PrintFailure(const char* errString) {
    Print("FAILURE IN SERVER: /cmd %s %s\n", Cmd::cmdName(), errString);
  }

  sc_msg_iter oscArgs() {
    return sc_msg_iter(oscDataSize, oscData);
  }

  BaseAsyncCmd() = delete;
  static Cmd* alloc(sc_msg_iter* args, World* world=nullptr) {
    size_t oscDataSize = args->remain();
    size_t dataSize = sizeof(Cmd) + oscDataSize;
    // Print("allocating data size: %d\n", dataSize);

    // this is just malloc: we are always passing world=nullptr
    Cmd* cmdData = (Cmd*) (world ? RTAlloc(world, dataSize) : NRTAlloc(dataSize));
    if (cmdData == nullptr) {
      Print("FAILURE IN SERVER %s: msg data alloc failed.\n", Cmd::cmdName());
      return nullptr;
    }
    cmdData->oscDataSize = oscDataSize;
    memcpy(cmdData + 1, args->data + args->size - args->remain(), oscDataSize);
    cmdData->oscData = reinterpret_cast<const char*>(cmdData + 1);

    return cmdData;
  }

  // static void rtFree(World* world, void* data) { RTFree(world, data); }
  static void nrtFree(World*, void* data) { NRTFree(data); }

  static void asyncCmd(World* world, void* inUserData, sc_msg_iter* args, void* replyAddr) {
    const char* cmdName = Cmd::cmdName();
    // Print("nn async cmd name: %s\n", cmdName);
    Cmd* data = Cmd::alloc(args, nullptr);
    if (data == nullptr) return;
    DoAsynchronousCommand(
      world, replyAddr, cmdName, data,
      Cmd::stage2, // stage2 is non real time
      Cmd::stage3, // stage3: RT (completion msg performed if true)
      Cmd::stage4, // stage4: NRT (sends /done if true)
      nrtFree, 0, 0);
  }

  static void define() {
    DefinePlugInCmd(Cmd::cmdName(), Cmd::asyncCmd, nullptr);
  }

  static bool stage2(World*, void*) { return true; }
  static bool stage3(World*, void*) { return true; }
  static bool stage4(World*, void*) { return true; }
};

struct NNLoadCmd : BaseAsyncCmd<NNLoadCmd> {

  static const char* cmdName() { return "/nn_load"; }
  // static constexpr const char* cmdName = "/nn_load";

  static bool stage2(World* world, void* inData) {
    auto cmdData = (NNLoadCmd*) inData;
    sc_msg_iter args = cmdData->oscArgs();
    const int id = args.geti(-1);
    const char* path = args.gets();
    const char* filename = args.gets("");
    if (path == 0) {
      cmdData->PrintFailure("needs a path to a .ts file");
      return false;
    }

    // Print("nn_load: idx %d path %s\n", id, path);
    auto model = (id == -1) ? gModels.load(path) : gModels.load(id, path);
    if (model == nullptr) {
      char errMsg[64 + strlen(path)]; sprintf(errMsg, "can't load model at %s", path);
      cmdData->PrintFailure(errMsg);
      return false;
    }

    if (strlen(filename) > 0) {
      bool success = model->dumpInfo(filename);
      if (!success) {
        char errMsg[64 + strlen(filename)]; sprintf(errMsg, "can't write file '%s'", filename);
        cmdData->PrintFailure(errMsg);
        return false;
      }
    }

    return true;
  }
};

// /cmd /nn_query str
struct NNQueryCmd : BaseAsyncCmd<NNQueryCmd> {

  static const char* cmdName() { return "/nn_query"; }

  static bool stage2(World* world, void* inData) {
    auto cmdData = (NNQueryCmd*) inData;
    sc_msg_iter args = cmdData->oscArgs();
    const int modelIdx = args.geti(-1);
    const char* outFile = args.gets("");

    bool writeToFile = strlen(outFile) > 0;

    // modelIdx < 0 is dumpAll
    if (modelIdx < 0) {
      if (writeToFile) {
        return gModels.dumpAllInfo(outFile);
      } else {
        gModels.printAllInfo();
      }
      return true;
    }

    const auto model = gModels.get(static_cast<unsigned short>(modelIdx), true);
    if (model) {
      if (writeToFile) {
        return model->dumpInfo(outFile);
      } else {
        model->printInfo();
      }
    }
    return true;
  }
};

// /nn_unload i
struct NNUnloadCmd : BaseAsyncCmd<NNUnloadCmd> {
public:
  static const char* cmdName() { return "/nn_unload"; }

  static bool stage2(World* world, void* inData) {
    auto cmdData = (NNUnloadCmd*) inData;
    sc_msg_iter args = cmdData->oscArgs();
    int id = args.geti(-1);
    gModels.unload(id);
    return true;
  }
};

// /cmd /nn_version
void nn_print_version(World*, void*, sc_msg_iter*, void*) {
  Print("nn.ar version %s compiled for SuperCollider %s with libtorch %s\n",
        NNAR_VERSION, SC_VERSION, TORCH_VERSION);
}

// // /cmd /nn_warmup int int
// struct NNWarmupCmd : BaseAsyncCmd<NNUnloadCmd> {
// public:
//   static const char* cmdName() { return "/nn_warmup"; }
//
//   static bool stage2(World* world, void* inData) {
//     auto cmdData = (NNWarmupCmd*) inData;
//     int modelIdx = cmdData->oscArgs->geti(-1);
//     int methodIdx = cmdData->oscArgs->geti(-1);
//
//    if (modelIdx < 0) {
//       // Print("nn_warmup: invalid model index %d\n", modelIdx);
//       const char errMsg[256];
//       sprintf(errMsg, "invalid model index %d", id);
//       cmdData->PrintFailure(errMsg);
//       return true;
//    }
//    const auto model = gModels.get(static_cast<unsigned short>(modelIdx), true);
//    if (model) {
//      if (methodIdx < 0) {
//        // warmup all methods */
//        for(auto method: model->m_methods) model->warmup_method(&method);
//      } else {
//        auto method = model->getMethod(methodIdx, true);
//        if (method) model->warmup_method(method);
//      }
//    }
//    return true;
//   }
// };


void definePlugInCmds() {
  NNLoadCmd::define();
  NNUnloadCmd::define();
  NNQueryCmd::define();
  DefinePlugInCmd("/nn_version", nn_print_version, nullptr);
  // NNWarmupCmd::define();
}

} // namespace NN::Cmd

