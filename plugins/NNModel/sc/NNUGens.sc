NNUGen : MultiOutUGen {

	*ar { |modelIdx, methodIdx, bufferSize, numOutputs, inputs, warmup, debug|
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
