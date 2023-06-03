NNModel {

	classvar models;
	var <server, <key;
	var <idx, <path, <minBufferSize, <methods, <settings;

	*initClass { models = IdentityDictionary[] }

	*get { |key| ^models[key.asSymbol] }

	*new { |key| ^this.get(key) }

	*load { |key, path, server(Server.default)|
		var model = this.get(key);
		model ?? {
			model = super.newCopyArgs(server, key);
			models[key] = model;
		};
		model.load(path);
		^model;
	}

	load { |path|

		if (server.serverRunning.not) { Error("server not running").throw };

		forkIfNeeded {
			var infoFile = PathName.tmp +/+ "nn-sc-%.json".format(UniqueID.next());
			var infoJson;
			server.sync(bundles: [
				["/cmd", "/nn_load", this.key, path, infoFile]
			]);
			infoJson = File.readAllString(infoFile).parseJSON;
			this.prParseInfoJson(infoJson);
			File.delete(infoFile);
		}
	}

	set { |settingName, value|

		var settingIdx = settings.indexOf(settingName.asSymbol);
		settingIdx ?? {
			Error("NNModel(%): setting % not found. Settings: %"
				.format(this.key, settingName, settings)).throw;
		};
		forkIfNeeded {
			server.sync(bundles: [
				["/cmd", "/nn_set", this.key, settingIdx, value]
			]);
		}
	}

	method { |name|
		methods ?? { Error("NNModel % has no methods.".format(key)).throw };
		^methods.detect { |m| m.name == name };
	}

	describe {
		"NNModel(%)".format(key).postln;
		"path: %".format(path).postln;
		"minBufferSize: %".format(minBufferSize).postln;
		methods.do { |m|
			"- method %: % in, % out".format(m.name, m.inDim, m.outDim).postln;
		}
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
		settings = json["settings"].collect(_.asSymbol)
	}

	printOn { |stream|
		stream << "%(%, %)%".format(this.class.name, key, minBufferSize, methods.collect(_.name));
	}
}

NNModelMethod {
	var <name, <idx, <inDim, <outDim;

	*new { |...args| ^super.newCopyArgs(*args) }
	printOn { |stream|
		stream << "%(%: % in, % out)".format(this.class.name, name, inDim, outDim);
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
		if (inputs.size != method.inDim) {
			Error("NNModel(%): method % has % inputs, but was given %."
				.format(key, methodName, method.inDim, inputs.size)).throw
		};
		^this.new1('audio', model.idx, method.idx, bufferSize, *inputs)
			.initOutputs(method.outDim, 'audio');
	}
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}
}
