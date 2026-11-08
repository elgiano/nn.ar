NNUGen : MultiOutUGen {

	// enum UGenInputs { modelIdx=0, methodIdx, bufSize, warmup, debug, nBatches, active, inputs };
	// todo: clump batches
	*ar { |modelIdx, methodIdx, bufferSize, numOutputs, warmup, debug, nBatches, active, inputs|
		^this.new1('audio', modelIdx, methodIdx, bufferSize, warmup, debug, nBatches, active, *inputs)
			.initOutputs(numOutputs * nBatches, 'audio');
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

+NNModelMethod {
	ar { |inputs, bufferSize=(-1), warmup=0, debug=0, attributes(#[]), active=1|
		var attrParams, nBatches, outputs;
		inputs = inputs.asArray;


		case
		// flat list of inputs:
		{ inputs.rank == 1} {
			case
			{ inputs.size == this.numInputs } { nBatches = 1 }
			{ this.numInputs == 1 } { nBatches = inputs.size }
			{ inputs.size % this.numInputs == 0 } {
				// support flat list of inputs (e.g. for \elastic NodeProxies)
				nBatches = inputs.size.div(this.numInputs);
				inputs = inputs.clump(this.numInputs).lace;
			}
		}
		// nested lists:
		{ inputs.rank == 2 && {inputs.every{ |b| b.size == this.numInputs }} } {
			nBatches = inputs.size;
			// ugen wants interlaced latents:
			// e.g. inputs (a,b) with latents (0..2): a0, b0, a1, b1, a2, b2 
			inputs = inputs.lace;
		} {
			Error("NNModel: method % has % inputs, but was given %."
				.format(this.name, this.numInputs, inputs.shape)).throw
		};


		attrParams = Array(attributes.size);
		attributes.pairsDo { |attrName, attrValue|
			attrParams.add(model.attrIdx(attrName));
			attrParams.add(attrValue ?? 0);
		};

		outputs = NNUGen.ar(model.idx, idx, bufferSize, this.numOutputs, warmup, debug, nBatches, active, inputs ++ attrParams);
		// ugen outputs is not interlaced batches: clump
		// e.g. [a0, a1, b0, b1] -> clump to [[a0,a1], [b0,b1]]
		// note that if numOutputs is 1 [a0,b0] doesn't need to be clumped
		if (nBatches > 1 && {this.numOutputs > 1}) {
			outputs = outputs.clump(this.numOutputs);
		};
		^outputs
	}
}
