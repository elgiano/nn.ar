
// NNModel can be constructed only:
// - *load: by loading a model on the server
// - *read: by reading an info file (for NRT)

NNModel {

	var <server, <path, <idx, <info, <methods;
	var <isLoaded=false;

	*new { ^nil }

  minBufferSize { ^if (info.isNil) { nil } { info.minBufferSize } }
  attributes { ^if(info.isNil, nil, info.attributes) }
	attrIdx { |attrName|
		var attrs = this.attributes ?? { ^nil };
		^attrs.indexOf(attrName);
	}

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
			var infoID = UniqueID.next; 
			infoFile = PathName.tmp +/+ "nn-sc-" ++ infoID ++ ".yaml"
		};

		loadMsg = NN.loadMsg(id, path, infoFile);

		model = super.newCopyArgs(server);

    forkIfNeeded {
      server.sync(bundles: [loadMsg]);
      // server writes info file: read it
      protect { 
        model.initFromFile(infoFile);
				ServerBoot.add(model, server);
        action.(model)
      } {
        File.delete(infoFile);
      }
    };

		^model;
	}

	doOnServerBoot {
		server.sendMsg(*this.loadMsg)
	}

  initFromFile { |infoFile|
    var info = NNModelInfo.fromFile(infoFile);
    this.initFromInfo(info);
    NN.prCacheInfo(info);
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

	get { |attributeName, action|
		{
			NNGet.kr(this.idx, attributeName)
		}.loadToFloatArray(server.options.blockSize / server.sampleRate, server) { |v|
			action.(v.last)
		}
	}

	setMsg { |attributeName, value|
		var attrIdx = this.attrIdx(attributeName.asSymbol) ?? {
			Error("NNModel(%): attribute % not found. Attributes: %"
				.format(this.key, attributeName, this.attributes)).throw;
		};
		^NN.setMsg(this.idx, attrIdx, value)
	}
	set { |attributeName, value|
		var msg = this.setMsg(attributeName, value);
		this.prErrIfNoServer("dumpInfo");
		if (server.serverRunning.not) { Error("server not running").throw };
		forkIfNeeded { server.sync(bundles: [msg]) };
	}

	warmup {
		forkIfNeeded {
			server.sync(bundles: [NN.warmupMsg(this.idx, -1)]);
		}
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
		stream << "NNModel(" <<* [this.key, this.minBufferSize] << ")";
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
	var <idx, <path, <minBufferSize, <methods, <attributes;
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
		attributes = yaml["attributes"].collect(_.asSymbol) ?? { [] }
  }
}

NNModelMethod {
	var <model, <name, <idx, <numInputs, <numOutputs;

	*new { |...args| ^super.newCopyArgs(*args) }

  copyForModel { |model|
    ^this.class.newCopyArgs(model, name, idx, numInputs, numOutputs)
  }

	ar { |bufferSize=0, inputs|
		inputs = inputs.asArray;
		if (inputs.size != this.numInputs) {
			Error("NNModel: method % has % inputs, but was given %."
				.format(this.name, this.numInputs, inputs.size)).throw
		};
		^NNUGen.ar(model.idx, idx, bufferSize, this.numOutputs, inputs)
	}

	warmup {
		forkIfNeeded {
			model.server.sync(bundles: [NN.warmupMsg(model.idx, this.idx)]);
		}
	}

	printOn { |stream|
		stream << "%(%: % in, % out)".format(this.class.name, name, numInputs, numOutputs);
	}
}
