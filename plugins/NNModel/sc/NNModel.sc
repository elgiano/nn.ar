NNModel {

	classvar models;
	var <server, <key;
	var <idx, <path, <minBufferSize, <methods, <settings;

	*initClass { models = IdentityDictionary[] }

	*get { |key| ^models[key.asSymbol] }

	*new { |key| ^this.get(key) }

	*load { |key, path, server(Server.default)|
		var model = this.get(key);
		if (path.isKindOf(String).not) {
			Error("NNModel.load: path needs to be a string, got: %").format(path).throw
		};
		model ?? {
			model = super.newCopyArgs(server, key);
			models[key] = model;
		};
		model.load(path);
		^model;
	}

	*loadMsg { |key, path, infoFile|
		^["/cmd", "/nn_load", key, path, infoFile]
	}
	loadMsg { |path, infoFile|
		^this.class.loadMsg(this.key, path, infoFile)
	}
	load { |path|
		var infoFile, loadMsg;
		path = path.standardizePath;
		if (server.serverRunning.not) {
			Error("server not running").throw
		};
		if (File.exists(path).not) {
			Error("model file '%' not found".format(path)).throw
		};

		infoFile = PathName.tmp +/+ "nn-sc-%.json".format(UniqueID.next());
		loadMsg = this.loadMsg(path, infoFile);

		forkIfNeeded {
			var infoJson;
			server.sync(bundles: [loadMsg]);
			// server writes info file: read it
			if (File.exists(infoFile).not) {
				error("NNModel: can't load info file '%'".format(infoFile));
			} {
				infoJson = File.readAllString(infoFile).parseJSON;
				this.prParseInfoJson(infoJson);
				File.delete(infoFile);
			}
		}
	}

	*setMsg { |modelIdx, settingIdx, value|
		^["/cmd", "/nn_set", modelIdx, settingIdx, value.asString]
	}
	setMsg { |settingName, value|
		var settingIdx = settings.indexOf(settingName.asSymbol);
		settingIdx ?? {
			Error("NNModel(%): setting % not found. Settings: %"
				.format(this.key, settingName, settings)).throw;
		};
		^this.class.setMsg(this.idx, settingIdx, value)
	}
	set { |settingName, value|
		var msg = this.setMsg(settingName, value);
		if (server.serverRunning.not) { Error("server not running").throw };
		forkIfNeeded { server.sync(bundles: [msg]) };
	}

	get { |settingName, action|
		{
			NNGet.kr(this.key, settingName)
		}.loadToFloatArray(server.options.blockSize / server.sampleRate, server) { |v|
			action.(v.last)
		}
	}

	method { |name|
		methods ?? { Error("NNModel % has no methods.".format(key)).throw };
		^methods.detect { |m| m.name == name };
	}

	prParseInfoJson { |json|
		idx = json["idx"].asInteger;
		path = json["modelPath"];
		minBufferSize = json["minBufferSize"];
		methods = json["methods"].collect { |m, n|
			var name = m["name"].asSymbol;
			var inDim = m["inDim"].asInteger;
			var outDim = m["outDim"].asInteger;
			NNModelMethod(name, n, inDim, outDim);
		};
		settings = json["settings"].collect(_.asSymbol) ?? { [] }
	}

	describe {
		"NNModel(%)".format(key).postln;
		"path: %".format(path).postln;
		"minBufferSize: %".format(minBufferSize).postln;
		methods.do { |m|
			"- method %: % ins, % outs".format(m.name, m.numInputs, m.numOutputs).postln;
		}
	}

	printOn { |stream|
		stream << "%(%, %)%".format(this.class.name, key, minBufferSize, methods.collect(_.name));
	}
}

NNModelMethod {
	var <name, <idx, <numInputs, <numOutputs;

	*new { |...args| ^super.newCopyArgs(*args) }
	printOn { |stream|
		stream << "%(%: % in, % out)".format(this.class.name, name, numInputs, numOutputs);
	}
}

NN : MultiOutUGen {

	*ar { |key, methodName, bufferSize, inputs|
		var model, method, method_idx;
		model = NNModel(key) ?? { 
			Error("NN: model % not found".format(key)).throw
		};
		method = model.method(methodName) ?? { 
			Error("NNModel(%): method % not found".format(key, methodName)).throw
		};
		inputs = inputs.asArray;
		if (inputs.size != method.numInputs) {
			Error("NNModel(%): method % has % inputs, but was given %."
				.format(key, methodName, method.numInputs, inputs.size)).throw
		};
		^this.new1('audio', model.idx, method.idx, bufferSize, *inputs)
			.initOutputs(method.numOutputs, 'audio');
	}
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}

NNSet : UGen {

	*kr { |key, settingName, input|
		var model, settingIdx;
		model = NNModel(key) ?? { 
			Error("NNSet: model % not found".format(key)).throw
		};
		settingIdx = model.settings.indexOf(settingName.asSymbol) ?? {
			Error("NNSet(%): setting % not found. Settings: %"
				.format(key, settingName, model.settings)).throw;
		};
		^this.new1('control', model.idx, settingIdx, input)
	}
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

}

NNGet : UGen {

	*kr { |key, settingName|
		var model, settingIdx;
		model = NNModel(key) ?? { 
			Error("NNGet: model % not found".format(key)).throw
		};
		settingIdx = model.settings.indexOf(settingName.asSymbol) ?? {
			Error("NNGet(%): setting % not found. Settings: %"
				.format(key, settingName, model.settings)).throw;
		};
		^this.new1('control', model.idx, settingIdx)
	}
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

}
