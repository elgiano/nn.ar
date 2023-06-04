# NN

[nn_tilde](https://github.com/acids-rave/nn_tilde) adaptation for SuperCollider: load torchscripts for real-time audio processing.

### Description
It has most features of nn_tilde:
- interface for any available model method (e.g. forward, encode, decode)
- interface for getting and setting model attributes
- processes real-time at different buffer sizes, on separate threads
- loads models asynchronously on scsynth

- tested so far only with [RAVE](https://github.com/acids-rave/rave) (v1 and v2) and [msprior](https://github.com/acids-rave/nn_tilde) models
- tested so far only on CPU, on linux

```supercollider
// Example:
s.waitForBoot {
    // when in a Routine, this method waits until the model has loaded
    NNModel.load(\rave, "~/rave/ravemodel.ts");
    // when model has loaded, print a description of all methods and parameters
    NNModel(\rave).describe;
}

NNModel(\rave).methods;
// -> NNModelMethod(forward: 1 ins, 1 outs), NNModelMethod(encode: 1 ins, 8 outs), ...


// play
{ NN.ar(\rave, \forward, WhiteNoise.ar) }.play

NNModel.load(\msprior, "~/rave/msprior.ts");

{
    var in, latent, modLatent, prior, resynth;
    
    in = SoundIn.ar();
    latent = NN.ar(\rave, \encode, 2048, in);
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(MouseY.kr.exprange(0.1, 30)).range(-0.5, 0.5)
    };
    prior = NN.ar(\msprior, \forward, 2048, latent);
    resynth = NN.ar(\rave, \decode, 2048, prior.drop(-1);

    resynth
}.play;

NNModel(\msprior).settings;
// -> [ listen, temperature, learn_context, reset ]

// set via command
NNModel(\rave).set(\listen, false);
NNModel(\rave).set(\listen, true);
// set via UGen
{
    var trig = Dust.kr(MouseY.kr.exprange(0.1, 10));
    NNSet.kr(\msprior, \listen, ToggleFF.kr(trig));
}.play;

// get parameter
NNModel(\msprior).get(\temperature)
{ NNGet.kr(\msprior, \temperature).poll }.play;
```


### Requirements

- CMake >= 3.5
- SuperCollider source code
- [libtorch](https://pytorch.org/cppdocs/installing.html)

### Building

Clone the project:

    git clone https://github.com/elgiano/nn-supercollider
    cd nn-supercollider
    mkdir build
    cd build

Then, use CMake to configure and build it:

    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cmake --build . --config Release --target install

You may want to manually specify the install location in the first step to point it at your
SuperCollider extensions directory: add the option `-DCMAKE_INSTALL_PREFIX=/path/to/extensions`.

It's expected that the SuperCollider repo is cloned at `../supercollider` relative to this repo. If
it's not: add the option `-DSC_PATH=/path/to/sc/source`.

### Developing

The usual `regenerate` command was disabled because `CmakeLists.txt` needed to be manually edited to include libtorch.
