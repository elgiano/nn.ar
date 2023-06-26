NN {

	classvar rtModelStore, rtModelsInfo;
	*initClass {
    rtModelStore = IdentityDictionary[];
    // store model info by path
    rtModelsInfo = IdentityDictionary[]
	}

	*models {
		^if(this.isNRT, this.nrtModelStore, rtModelStore).values;
	}
	*model { |key|
		^if(this.isNRT, this.nrtModelStore, rtModelStore)[key];
	}
	*keyForModel { |model|
		^if(this.isNRT, this.nrtModelStore, rtModelStore).findKeyForValue(model);
	}
	*prPut { |key, model|
		if(this.isNRT, this.nrtModelStore, rtModelStore)[key] = model;
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
	// NN(\model, \method) -> NNModelMethod
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
      Error("NN.load: path needs to be a string, got: %").format(path).throw
    };
    if (model.isNil) {
      if (this.isNRT) {
        var info =  this.prGetCachedInfo(path) ?? {
          Error("NN.load (nrt): model info not found for %".format(path)).throw;
        };
        model = NNModel.fromInfo(info, this.nextModelID);
				this.prPut(key, model);
      } {
        model = NNModel.load(path, id, server, action: { |m|
					this.prPut(key, m);
					// call action after adding to registry: in case action needs key
					action.value(m);
				});
      };
    };
		if (this.isNRT) {
			server.sendMsg(*model.loadMsg);
		}
    ^model;
  }

	*describeAll { this.models.do(_.describe) }
	
	*dumpInfo { |outFile, server(Server.default)|
		forkIfNeeded {
			server.sync(bundles:[this.dumpInfoMsg(-1, outFile)])		
		}
	}

	*loadMsg { |id, path, infoFile|
		^["/cmd", "/nn_load", id, path.standardizePath, infoFile.standardizePath]
	}
	*setMsg { |modelIdx, attrIdx, value|
		^["/cmd", "/nn_set", modelIdx, attrIdx, value.asString]
	}
	*dumpInfoMsg { |modelIdx, outFile|
		^["/cmd", "/nn_query", modelIdx ? -1, outFile ? ""]
	}
	*warmupMsg { |modelId, methodId(-1)|
		^["/cmd", "/nn_warmup", modelId, methodId]
	}

}
