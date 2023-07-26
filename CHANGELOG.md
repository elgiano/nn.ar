# CHANGELOG

### v0.0.4-alpha
- NNUGen: allow for a custom number of warmup passes (on my setup with rave v2 models, 2 warmup passes work well to avoid initial stuttering)

### v0.0.3-alpha
changed implementation to independent per-UGen model instance
- NN.load: scsynth only loads model to read info, real loading is done in UGen
- attributes interface: now only in UGen, no more set and get methods
- added silent warmup pass option to UGen
- UGen interface: blockSize moved from first to second arg, first is inputs, added debug, warmup and attributes args

### v0.0.2-alpha
- updated backend from nn_tilde: using a looping thread
- don't wait for thread joins either in ::next nor in Dtor, for a smoother audio chain
- ringbuffer: use memcpy instead of loop

### v0.0.1-alpha
- cleaned interface and first NRT implementation
