NN {

	classvar rtModelStore, rtModelsInfo;
	*initClass {
    rtModelStore = IdentityDictionary[];
    // store model info by path
    rtModelsInfo = IdentityDictionary[]
	}

  *isNRT { ^currentEnvironment[\nn_nrt].notNil }
	*modelStore { ^if(this.isNRT) { this.nrtModelStore } { rtModelStore } }
	*models { ^this.modelStore.values }
	*model { |key| ^this.modelStore[key] }
  *modelsInfo { ^if(this.isNRT) { this.nrtModelsInfo } { rtModelsInfo }  }

  *cacheInfo { |info|
		var path = info.path.asSymbol;
    if (this.modelsInfo[path].notNil) {
      "NN: overriding cached info for '%'".format(path).warn;
    };
    this.modelsInfo[path] = info;
  }
	*getCachedInfo { |path| ^this.modelsInfo[path.standardizePath.asSymbol] }

  *nrtModelStore {
    if (this.isNRT.not) { ^nil };
    ^currentEnvironment[\nn_nrt].models;
  }

  *nrtModelsInfo {
    if (this.isNRT.not) { ^nil };
    ^currentEnvironment[\nn_nrt].modelsInfo;
  }

  *nextModelID {
    if (this.isNRT.not) { ^nil };
    ^currentEnvironment[\nn_nrt].modelAllocator.alloc;
  }

	*nrt { |infoFile, fn|
		^Environment[\nn_nrt -> (
      modelAllocator: ContiguousBlockAllocator(1024),
      modelsInfo: IdentityDictionary[],
      models: IdentityDictionary[]
    )].use { 
      NN.readInfo(infoFile);
      Server.default.makeBundle(false, fn)
    };	
	}

  *load { |key, path, id(-1), server(Server.default), action|
    var model = this.model(key);
    if (path.isKindOf(String).not) {
      Error("NN.load: path needs to be a string, got: %").format(path).throw
    };
    if (model.isNil) {
      if (this.isNRT) {
        var info =  this.getCachedInfo(path) ?? {
          Error("NN.load (nrt): model info not found for %".format(path)).throw;
        };
        model = NNModel.fromInfo(info, this.nextModelID);
      } {
        model = NNModel.load(path, id, server, action);
      };
      this.modelStore.put(key, model);
    };
		if (this.isNRT) {
			server.sendMsg(*model.loadMsg.postln);
		}
    ^model;
  }
	
	*readInfo { |infoFile|
		if (File.exists(infoFile).not) {
			Error("NNModel: can't load info file '%'".format(infoFile)).throw;
		} {
			var yaml = File.readAllString(infoFile).parseYAML;
			var models = yaml.collect { |modelInfo|
				var info = NNModelInfo.fromDict(modelInfo);
        this.cacheInfo(info);
			};
		}
	}

	// NN(\model) -> NNModel
	// NN(\model, \method) -> NNModelMethod
	*new { |key, methodName|
		var model = this.model(key) ?? { 
			Error("NNModel: model % not found".format(key)).throw;
		};
		if (methodName.isNil) { ^model };
		^model.method(methodName).postln ?? {
			Error("NNModel(%): method % not found".format(key, methodName)).throw
		};
	}

	*dumpInfo { |outFile, server(Server.default)|
		forkIfNeeded {
			server.sync(bundles:[this.dumpInfoMsg(-1, outFile)])		
		}
	}

	*loadMsg { |id, path, infoFile|
		path = path.standardizePath;
		^["/cmd", "/nn_load", id, path, infoFile]
	}
	*setMsg { |modelIdx, settingIdx, value|
		^["/cmd", "/nn_set", modelIdx, settingIdx, value.asString]
	}
	*dumpInfoMsg { |modelIdx, outFile|
		^["/cmd", "/nn_query", modelIdx ? -1, outFile ? ""]
	}

	*keyForModel { |model| ^this.modelStore.findKeyForValue(model) }
}

// NNModel can be constructed only:
// - *load: by loading a model on the server
// - *read: by reading an info file (for NRT)

NNModel {

	var <server, <path, <idx, <info, <methods;

	*new { ^nil }

  minBufferSize { ^if (info.isNil, nil, info.minBufferSize) }
  settings { ^if(info.isNil, nil, info.settings) }

	*load { |path, id(-1), server(Server.default), action|
		var loadMsg, infoFile, model;
		path = path.standardizePath;
		if (server.serverRunning.not) {
			Error("server not running").throw
		};
		if (File.exists(path).not) {
			Error("model file '%' not found".format(path)).throw
		};

		if (infoFile.isNil) {
			infoFile = PathName.tmp +/+ "nn-sc-%.yaml".format(UniqueID.next())
		};
		loadMsg = NN.loadMsg(id, path, infoFile);

		model = super.newCopyArgs(server);

    forkIfNeeded {
      server.sync(bundles: [loadMsg]);
      // server writes info file: read it
      protect { 
        model.initFromFile(infoFile);
        action.(model)
      } {
        File.delete(infoFile);
      }
    };

		^model;
	}

  initFromFile { |infoFile|
    var info = NNModelInfo.fromFile(infoFile);
    this.initFromInfo(info);
    NN.cacheInfo(info);
  }
  initFromInfo { |infoObj, overrideId|
    info = infoObj;
    path = info.path;
    idx = overrideId ? info.idx;
    methods = info.methods.collect { |m| m.copyForModel(this) }
  }
	
	*read { |infoFile, server(Server.default)|
		^super.newCopyArgs(server).initFromFile(infoFile);
	}
	*fromInfo { |info, overrideId, server(Server.default)|
		^super.newCopyArgs(server).initFromInfo(info, overrideId);
	}

	loadMsg { |newPath, infoFile|
		^NN.loadMsg(idx, newPath ? path, infoFile)
	}

	get { |settingName, action|
		{
			NNGet.kr(this.idx, settingName)
		}.loadToFloatArray(server.options.blockSize / server.sampleRate, server) { |v|
			action.(v.last)
		}
	}

	setMsg { |settingName, value|
		var settingIdx = this.settings.indexOf(settingName.asSymbol);
		settingIdx ?? {
			Error("NNModel(%): setting % not found. Settings: %"
				.format(this.key, settingName, this.settings)).throw;
		};
		^NN.setMsg(this.idx, settingIdx, value)
	}
	set { |settingName, value|
		var msg = this.setMsg(settingName, value);
		this.prErrIfNoServer("dumpInfo");
		if (server.serverRunning.not) { Error("server not running").throw };
		forkIfNeeded { server.sync(bundles: [msg]) };
	}

	dumpInfoMsg { |outFile| ^NN.dumpInfoMsg(this.idx, outFile) }
	dumpInfo { |outFile|
		var msg = this.dumpInfoMsg(outFile);
		this.prErrIfNoServer("dumpInfo");
		if (server.serverRunning.not) { Error("server not running").throw };
		forkIfNeeded { server.sync(bundles:[msg]) }
	}

	method { |name|
    var method;
		this.methods ?? { Error("NNModel % has no methods.".format(this.key)).throw };
		^this.methods.detect { |m| m.name == name };
	}

	describe {
		"\n*** NNModel(%)".format(this.key).postln;
		"path: %".format(this.path).postln;
		"minBufferSize: %".format(this.minBufferSize).postln;
		this.methods.do { |m|
			"- method %: % ins, % outs".format(m.name, m.numInputs, m.numOutputs).postln;
		};
		"".postln;
	}

	printOn { |stream|
		stream << "NNModel(%, %)%".format(this.key, this.minBufferSize, this.methods.collect(_.name));
	}

	prErrIfNoServer { |funcName|
		if (server.isNil) {
			Error("%: NNModel(%) is not bound to a server, can't dumpInfo. Is it a NRT model?"
				.format(funcName, this.key)).throw
		};
	}

	key { ^NN.keyForModel(this) }

}

NNModelInfo {
	var <idx, <path, <minBufferSize, <methods, <settings;
  *new {}

  *fromFile { |infoFile|
    if (File.exists(infoFile).not) {
      Error("NNModelInfo: can't load info file '%'".format(infoFile)).throw;
    } {
      var yaml = File.readAllString(infoFile).parseYAML[0];
      ^super.new.initFromDict(yaml)
    }
  }
	*fromDict { |infoDict|
		^super.new.initFromDict(infoDict);
	}
  initFromDict { |yaml|
		idx = yaml["idx"].asInteger;
		path = yaml["modelPath"];
		minBufferSize = yaml["minBufferSize"].asInteger;
		methods = yaml["methods"].collect { |m, n|
			var name = m["name"].asSymbol;
			var inDim = m["inDim"].asInteger;
			var outDim = m["outDim"].asInteger;
			NNModelMethod(nil, name, n, inDim, outDim);
		};
		settings = yaml["settings"].collect(_.asSymbol) ?? { [] }
  }
}

NNModelMethod {
	var <model, <name, <idx, <numInputs, <numOutputs;

	*new { |...args| ^super.newCopyArgs(*args) }

  copyForModel { |model|
    ^this.class.newCopyArgs(model, name, idx, numInputs, numOutputs)
  }

	ar { |bufferSize, inputs|
		inputs = inputs.asArray;
		if (inputs.size != this.numInputs) {
			Error("NNModel: method % has % inputs, but was given %."
				.format(this.name, this.numInputs, inputs.size)).throw
		};
		^NNUGen.ar(model.idx, idx, bufferSize, this.numOutputs, inputs)
	}

	printOn { |stream|
		stream << "%(%: % in, % out)".format(this.class.name, name, numInputs, numOutputs);
	}
}
