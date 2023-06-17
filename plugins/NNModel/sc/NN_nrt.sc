+NN {

	*nrt { |infoFile, makeBundleFn|
		^Environment[\nn_nrt -> (
      modelAllocator: ContiguousBlockAllocator(1024),
      modelsInfo: IdentityDictionary[],
      models: IdentityDictionary[]
    )].use { 
      NN.prReadInfoFile(infoFile);
      Server.default.makeBundle(false, makeBundleFn)
    };	
	}

	*prReadInfoFile { |infoFile|
		if (File.exists(infoFile).not) {
			Error("NNModel: can't load info file '%'".format(infoFile)).throw;
		} {
			var yaml = File.readAllString(infoFile).parseYAML;
			var models = yaml.collect { |modelInfo|
				var info = NNModelInfo.fromDict(modelInfo);
        this.prCacheInfo(info);
			};
		}
	}

  *isNRT { ^currentEnvironment[\nn_nrt].notNil }

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
}
