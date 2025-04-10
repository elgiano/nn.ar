# CHANGELOG

### v0.0.8-alpha
- add NN.debug and /nn_debug for more verbose messages
- add docs for /nn_version and mac/win install

### v0.0.7-alpha
- NNUGen: support flat list of multichannel inputs
- nn_load: fixed a crash when loading again after load failed
- updated macOS deQuarantine
- fix memory issue for async plugin commands
- add cmd /nn_version and NN.pluginVersion
- more detailed errors on NN.load

### v0.0.6-alpha
- Fixed a bug with multi-channel output, affecting \encode methods: unlacing was not needed

### v0.0.5-alpha
- Multichannel batch processing: multiple inputs will be processed *by the same model* as parallel batches
- only scsynth checks if a model is already loaded before (re-)loading it

### v0.0.4-alpha
- NNUGen: allow for a custom number of warmup passes (on my setup with rave v2 models, 2 warmup passes work well to avoid initial stuttering)
- NN: removed automatic model reload on server reboot, in favor of a .reload method

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
