TestNN : UnitTest {
	var server;
	classvar <testModelPath;
	classvar <testModelErrPath;

	*initClass {
		passVerbosity = UnitTest.brief;
		testModelPath = "nnar_test_model.ts".resolveRelative;
		testModelErrPath = "nnar_test_invalid_model.ts".resolveRelative;
	}

	setUp {
		server = Server(this.class.name);
		// blockSize needs to be equal to model blockSize
		server.options.blockSize = 16;
		server.bootSync;
	}

	tearDown {
		server.quit;
		server.remove;
	}


	// Async helper
	loadTestModelSync { |timeout=5|
		var cond = CondVar(), loaded = nil;

		NN.load(\test, testModelPath, server: server, action: {
			loaded = true; cond.signalOne()
		});
		cond.waitFor(timeout) { loaded.notNil };

		if (loaded.not) {
			Error("TestNN: can't load test model at %".format(testModelPath)).throw;
		};
		^NN(\test);
	}

	awaitSynthOutput { |synthFn, runtimeSamples=(server.options.blockSize), timeout=1|
		var cond = CondVar(), synthOutput = nil;
		var runtimeSeconds = runtimeSamples / server.sampleRate;
		// allow extra timeout for longer tests (e.g. testFFT with big memSize)
		if (runtimeSeconds > timeout) {
			timeout = runtimeSeconds + timeout
		};
		synthFn.loadToFloatArray(runtimeSeconds, server) { |out|
			synthOutput = out;
			cond.signalOne();
		};
		cond.waitFor(timeout) { synthOutput.notNil };
		^synthOutput;
	}

	test_load_fail {
		var err = nil;
		try {
			NN.load(\fail, "thismodeldoesnotexist.ts", server: server)
		} { |e| err = e }; 
		this.assert(err.notNil, 
			"sclang should throw an error when loading a non-existing file");

		err = nil;
		try {
			NN.load(\fail, testModelErrPath, server: server)
		} { |e| err = e }; 
		this.assert(err.notNil, 
			"sclang should throw an error when loading an invalid model");
	}

	test_load {
		var m = this.loadTestModelSync();	

		this.assertEquals(m.info.sampleRate, 44100,
			"should read model's sample rate");
		this.assertEquals(m.info.minBufferSize, 16,
			"should read model's minBufferSize");

		this.assert(m.method(\encode).notNil,
			"should read model's \\encode method");
		this.assertEquals(m.method(\encode).numInputs, 1,
			"should read method \\encode's numInputs");
		this.assertEquals(m.method(\encode).numOutputs, 8,
			"should read method \\encode's numOutputs");

		this.assert(m.method(\decode).notNil,
			"should read model's \\decode method");
		this.assertEquals(m.method(\decode).numInputs, 8,
			"should read method \\decode's numInputs");
		this.assertEquals(m.method(\decode).numOutputs, 1,
			"should read method \\decode's numOutputs");

		this.assert(m.method(\forward).notNil,
			"should read model's \\forward method");
		this.assertEquals(m.method(\forward).numInputs, 1,
			"should read method \\forward's numInputs");
		this.assertEquals(m.method(\forward).numOutputs, 1,
			"should read method \\forward's numOutputs");

		this.assertEquals(m.attributes[0], \test_attr,
				"should read model's attribute \\test_attr");
	}

	test_process {
		var m = this.loadTestModelSync();	
		var expected, actual;

		expected = [ 
			0.72988921403885, 0.39151978492737, 0.53745055198669, 0.65946221351624,
			0.19778919219971, -0.13562947511673, 0.5163933634758, 0.606116771698,
			0.32432875037193, -0.60026001930237, 0.38142707943916, 0.70655179023743,
			-0.24246397614479, 0.70271861553192, 0.4327227473259, 0.32896250486374
		];
		actual = this.awaitSynthOutput {
			NN(\test, \forward).ar(DC.ar(1), 0)
		};
		this.assertArrayFloatEquals(actual, expected, 
			"should produce expected results for \\forward", within:1e-10);

		expected = [
			0.2347609102726, 0.5710871219635, -0.93283587694168, 1.2672786712646,
			0.28386980295181, -0.02412423491478, -0.76934659481049, 0.29714974761009
		];
		actual = this.awaitSynthOutput {
			NN(\test, \encode).ar(DC.ar(1), 0)
		};
		this.assertArrayFloatEquals(actual, expected, 
			"should produce expected results for \\encode", within:1e-10);

		expected = [
			0.64241540431976, 0.11089354753494, 0.51640564203262, 0.050253599882126,
			0.38414990901947, -0.10745593905449, -0.5528444647789, 0.8644112944603, 
			0.37826299667358, 0.29239726066589, -0.46496778726578, 0.4191926419735,
			0.37278172373772, 0.37453624606133, 0.35215708613396, 0.17822727560997
		];
		actual = this.awaitSynthOutput {
			NN(\test, \decode).ar(DC.ar(1!8), 0)
		};
		this.assertArrayFloatEquals(actual, expected, 
			"should produce expected results for \\decode", within:1e-10);
	}
}
