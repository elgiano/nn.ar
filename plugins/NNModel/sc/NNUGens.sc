NNUGen : MultiOutUGen {

  // enum UGenInputs { modelIdx=0, methodIdx, bufSize, warmup, debug, inputs };
	*ar { |modelIdx, methodIdx, bufferSize, numOutputs, warmup, debug, inputs|
		^this.new1('audio', modelIdx, methodIdx, bufferSize, warmup, debug, *inputs)
			.initOutputs(numOutputs, 'audio');
	}
	
	checkInputs {
		// modelIdx, methodIdx and bufferSize are not modulatable
		['modelIdx', 'methodIdx', 'bufferSize'].do { |name, n|
			if (inputs[n].rate != \scalar) {
				^": '%' is not modulatable. Got: %.".format(name, inputs[n]);	
			}
		}
		^this.checkValidInputs;
	}
}

+ NNModelMethod {

	ar { |inputs, bufferSize=0, warmup=0, debug=0, attributes(#[])|
		var attrParams;
		inputs = inputs.asArray;
		if (inputs.size != this.numInputs) {
			Error("NNModel: method % has % inputs, but was given %."
				.format(this.name, this.numInputs, inputs.size)).throw
		};

		attrParams = Array(attributes.size);
		attributes.pairsDo { |attrName, attrValue|
			attrParams.add(model.attrIdx(attrName));
			attrParams.add(attrValue ?? 0);
		};

		^NNUGen.ar(model.idx, idx, bufferSize, this.numOutputs, warmup, debug, inputs ++ attrParams)
	}
}
