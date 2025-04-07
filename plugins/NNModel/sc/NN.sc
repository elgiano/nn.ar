NN {
	classvar rtModelStore, rtModelsInfo, <>tmpPath;
	*initClass {
		tmpPath = PathName.tmp;
		rtModelStore = IdentityDictionary[];
		// store model info by path
		rtModelsInfo = IdentityDictionary[];
	}

	// NN keeps track of all loaded models
	*prModelStore {
		^if(this.isNRT, this.nrtModelStore, rtModelStore);
	}
	*models { ^this.prModelStore.values }
	*model { |key| ^this.prModelStore[key] }
	*prPutModel { |key, model| this.prModelStore[key] = model }
	*prDeleteModel { |key| this.prModelStore.removeAt(key) }
	*keyForModel { |model|
		^this.prModelStore.findKeyForValue(model);
	}

	*prCacheInfo { |info|
		var cache = if(this.isNRT, this.nrtModelsInfo, rtModelsInfo);
		var path = info.path.asSymbol;
		if (cache[path].notNil) {
			"NN: overriding cached info for '%'".format(path).warn;
		};
		cache[path] = info;
	}
	*prGetCachedInfo { |path|
		^if(this.isNRT, this.nrtModelsInfo, rtModelsInfo)[path.standardizePath.asSymbol]
	}

	// NN(\model) -> NNModel
	// NN(\model, \method) -> NNUGenFactory
	*new { |key, methodName|
		var model = this.model(key) ?? { 
			Error("NNModel: model '%' not found".format(key)).throw;
		};
		if (methodName.isNil) { ^model };
		^model.method(methodName) ?? {
			Error("NNModel(%): method '%' not found".format(key, methodName)).throw
		};
	}

	*load { |key, path, id(-1), server(Server.default), action|
		var model = this.model(key);
		if (path.isKindOf(String).not) {
			Error("NN.load: path needs to be a string, got: %".format(path)).throw
		};
		if (this.isNRT) {
			var info =  this.prGetCachedInfo(path) ?? {
				Error("NN.load (nrt): model info not found for %".format(path)).throw;
			};
			model = NNModel.fromInfo(info, this.nextModelID);
			this.prPutModel(key, model);
		} {
			model = NNModel.load(path, id, server, action: { |m|
				this.prPutModel(key, m);
				// call action after adding to registry: in case action needs key
				action.value(m);
			});
		};
		if (this.isNRT) {
			server.sendMsg(*model.loadMsg);
		}
		^model;
	}

	*describeAll { this.models.do(_.describe) }

	*dumpInfo { |outFile, server(Server.default)|
		server.sendMsg(*this.dumpInfoMsg(-1, outFile))
	}

	*loadMsg { |id, path, infoFile|
		path = path !? { path.standardizePath };
		infoFile = infoFile !? { infoFile.standardizePath };
		^["/cmd", "/nn_load", id, path, infoFile]
	}
	*dumpInfoMsg { |modelIdx, outFile|
		^["/cmd", "/nn_query", modelIdx ? -1, outFile ? ""]
	}

}
