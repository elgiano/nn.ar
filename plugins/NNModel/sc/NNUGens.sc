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
