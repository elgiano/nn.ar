NNUGen : MultiOutUGen {

	*ar { |modelIdx, methodIdx, bufferSize, numOutputs, inputs|
		^this.new1('audio', modelIdx, methodIdx, bufferSize, *inputs)
			.initOutputs(numOutputs, 'audio');
	}
	
	checkInputs {
		// modelIdx, methodIdx and bufferSize are not modulatable
		['modelIdx', 'methodIdx', 'bufferSize'].do { |name, n|
			if (inputs[n].rate != \scalar) {
				^": input % is not modulatable".format(name);	
			}
		}
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
