# CHANGELOG

### v0.0.2-alpha
- updated backend from nn_tilde: using a looping thread
- don't wait for thread joins either in ::next nor in Dtor, for a smoother audio chain
- ringbuffer: use memcpy instead of loop

### v0.0.1-alpha
- cleaned interface and first NRT implementation
