NNNRTEnv {
	var <modelAllocator, <modelsInfo, <models;
	*new { ^super.new.init }
	init {
		modelAllocator = ContiguousBlockAllocator(1024);
		modelsInfo = IdentityDictionary[];
		models = IdentityDictionary[];
	}
}

+NN {

	*nrt { |infoFile, makeBundleFn|
		^Environment[\nn_nrt -> NNNRTEnv()].use { 
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

	*isNRT { ^currentEnvironment[\nn_nrt].isKindOf(NNNRTEnv) }

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

+ NNModel {

	nrtResynth { |bufPath, dstPath, blockSize=0|
		var startTime = Date.getDate.rawSeconds;
		var sampleRate, nch, duration;
		SoundFile.use(bufPath) { |sf|
			sampleRate = sf.sampleRate;
			nch = sf.numChannels;
			duration = sf.duration;
		};

		Score([
			[0.0, this.loadMsg],
			[0.0, ["/d_recv", SynthDef(\resynth) { |out=0|
				Out.ar(out, SoundIn.ar((0..nch)).collect { |ch|
					this.method(\forward).ar(ch, bufferSize: blockSize)
				})
				}.asBytes]],
			[0.0, Synth.basicNew(\resynth).newMsg],
			[duration + (blockSize / sampleRate)]
			]).recordNRT(
		inputFilePath: bufPath,
				outputFilePath: dstPath,
				headerFormat: "wav",
				sampleRate: sampleRate,
				sampleFormat: "float",
				action: { "done in %".format(Date.getDate.rawSeconds - startTime).postln },
				options: ServerOptions()
					.numInputBusChannels_(nch).numOutputBusChannels_(nch)
		)
	}
}
