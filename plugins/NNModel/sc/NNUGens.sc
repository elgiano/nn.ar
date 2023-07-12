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

	*kr { |key, attributeName, input|
		var model, attrIdx;
		model = NN(key) ?? { 
			Error("NNSet: model % not found".format(key)).throw
		};
		attrIdx = model.attrIdx(attributeName.asSymbol) ?? {
			Error("NNSet(%): attribute % not found. Attributes: %"
				.format(key, attributeName, model.attributes)).throw;
		};
		^this.new1('control', model.idx, attrIdx, input)
	}
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

}

NNGet : UGen {

	*kr { |key, attributeName|
		var model, attrIdx;
		model = NN(key) ?? { 
			Error("NNGet: model % not found".format(key)).throw
		};
		attrIdx = model.attrIdx(attributeName.asSymbol) ?? {
			Error("NNGet(%): attribute % not found. Attributes: %"
				.format(key, attributeName, model.attributes)).throw;
		};
		^this.new1('control', model.idx, attrIdx)
	}
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

}
