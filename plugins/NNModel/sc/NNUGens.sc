NNUGenFactory {

	classvar <idAllocator;
	classvar <currentSynthDef;

	var <ugenIdx, <model, <method;

	*initClass { this.reset }
	*reset {
		idAllocator = ContiguousBlockAllocator(512);
	}

	*nextUGenId {
		var synthDef = UGen.buildSynthDef ?? {
			Error("Can't create NN UGens outside of a SynthDef").throw
		};
		if (synthDef != this.currentSynthDef) {
			currentSynthDef = synthDef;
			this.reset
		};
		^idAllocator.alloc
	}

	*new { |model, method|
		var ugenIdx = this.nextUGenId;
		^super.newCopyArgs(ugenIdx, model, method);
	}

	ar { |bufferSize=0, inputs|
		inputs = inputs.asArray;
		if (inputs.size != method.numInputs) {
			Error("NNModel: method % has % inputs, but was given %."
				.format(method.name, method.numInputs, inputs.size)).throw
		};
		^NNUGen.new1('audio', ugenIdx, model.idx, method.idx, bufferSize, *inputs)
			.initOutputs(method.numOutputs, 'audio');
	}

	set { |attributeName, input|
		var attrIdx = model.attrIdx(attributeName.asSymbol) ?? {
			Error("NNSet(%): attribute % not found. Attributes: %"
				.format(model.key, attributeName, model.attributes)).throw;
		};
		^NNSet.new1('control', ugenIdx, attrIdx, input)
	}

	get { |attributeName|
		var attrIdx = model.attrIdx(attributeName.asSymbol) ?? {
			Error("NNSet(%): attribute % not found. Attributes: %"
				.format(model.key, attributeName, model.attributes)).throw;
		};
		^NNGet.new1('control', ugenIdx, attrIdx)
	}
	
}

NNUGen : MultiOutUGen {

	*ar { |ugenIdx, modelIdx, methodIdx, bufferSize, numOutputs, inputs|
		^this.new1('audio', ugenIdx, modelIdx, methodIdx, bufferSize, *inputs)
			.initOutputs(numOutputs, 'audio');
	}
	
	checkInputs {
		// modelIdx, methodIdx and bufferSize are not modulatable
		['ugenIdx', 'modelIdx', 'methodIdx', 'bufferSize'].do { |name, n|
			if (inputs[n].rate != \scalar) {
				^": input % is not modulatable. Got: %.".format(name, inputs[n]);	
			}
		}
		^this.checkValidInputs;
	}
}

NNSet : UGen {

	// *kr { |ugenIdx, key, attributeName, input|
	// 	var model, attrIdx;
	// 	model = NN(key) ?? { 
	// 		Error("NNSet: model % not found".format(key)).throw
	// 	};
	// 	attrIdx = model.attrIdx(attributeName.asSymbol) ?? {
	// 		Error("NNSet(%): attribute % not found. Attributes: %"
	// 			.format(key, attributeName, model.attributes)).throw;
	// 	};
	// 	^this.new1('control', ugenIdx, model.idx, attrIdx, input)
	// }
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

}

NNGet : UGen {

	// *kr { |key, attributeName|
	// 	var model, attrIdx;
	// 	model = NN(key) ?? { 
	// 		Error("NNGet: model % not found".format(key)).throw
	// 	};
	// 	attrIdx = model.attrIdx(attributeName.asSymbol) ?? {
	// 		Error("NNGet(%): attribute % not found. Attributes: %"
	// 			.format(key, attributeName, model.attributes)).throw;
	// 	};
	// 	^this.new1('control', model.idx, attrIdx)
	// }
	
	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

}
