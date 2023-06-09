# `NN.ar()`

[nn_tilde](https://github.com/acids-ircam/nn_tilde) adaptation for SuperCollider: load torchscripts for real-time audio processing.

### Description
It has most features of nn_tilde:
- interface for any available model method (e.g. forward, encode, decode)
- interface for getting and setting model attributes
- processes real-time at different buffer sizes, on separate threads
- loads models asynchronously on scsynth

- tested so far only with [RAVE](https://github.com/acids-ircam/rave) (v1 and v2) and [msprior](https://github.com/caillonantoine/msprior) models
- tested so far only on CPU, on linux

```supercollider
// Example:
// 1. load
s.waitForBoot {
    // when in a Routine, this method waits until the model has loaded
    NN.load(\rave, "~/rave/ravemodel.ts");
    // when model has loaded, print a description of all methods and attributes
    NN(\rave).describe;
}

NN(\rave).methods;
// -> NNModelMethod(forward: 1 ins, 1 outs), NNModelMethod(encode: 1 ins, 8 outs), ...


// 2. play
{ NN(\rave, \forward).ar(1024, WhiteNoise.ar) }.play;

NN.load(\msprior, "~/rave/msprior.ts", action: _.describe);

{
    var in, latent, modLatent, prior, resynth;

    in = SoundIn.ar();
    latent = NN(\rave, \encode).ar(2048, in);
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(MouseY.kr.exprange(0.1, 30)).range(-0.5, 0.5)
    };
    prior = NN(\msprior, \forward).ar(2048, latent);
    resynth = NN(\rave, \decode).ar(2048, prior.drop(-1);

    resynth
}.play;

NN(\msprior).attributes;
// -> [ listen, temperature, learn_context, reset ]

// set attribute via command
NN(\msprior).set(\listen, false);
NN(\msprior).set(\listen, true);
// set via UGen
{
    var trig = Dust.kr(MouseY.kr.exprange(0.1, 10));
    NNSet.kr(\msprior, \listen, ToggleFF.kr(trig));
}.play;

// get attribute
NN(\msprior).get(\temperature) { |val| val.postln } 

{ NNGet.kr(\msprior, \temperature).poll }.play;
```

### Installation

#### Download a pre-built release

- Download the latest release for your OS on the [Releases page](https://github.com/elgiano/nn.ar/releases).
- Extract the archive and copy the `nn.ar` folder to your SuperCollider Extensions folder

Note for mac users: binaries are not signed, so you can run the following in SuperCollider to bypass macos security complaints:
```supercollider
runInTerminal("xattr -d -r com.apple.quarantine %".format(Platform.userExtensionDir +/+ "nn.ar"))
runInTerminal("xattr -d -r com.apple.quarantine %".format(Platform.userExtensionDir +/+ "nn.ar/ignore"))
```

#### Building from source
If you compile SuperCollider from source, or if you want to enable optimizations specific to your machine, you need to build this extension yourself.

Build requirements:

- CMake >= 3.5
- SuperCollider source code
- [libtorch](https://pytorch.org/cppdocs/installing.html)

Clone the project:

    git clone https://github.com/elgiano/nn-supercollider
    cd nn-supercollider
    mkdir build
    cd build

Then, use CMake to configure:

    cmake .. -DCMAKE_BUILD_TYPE=Release

Libtorch is found automatically if installed system-wise. If you followed the official install instruction for libtorch (link above), you need to add it to CMAKE_PREFIX_PATH:

    cmake .. -DCMAKE_PREFIX_PAH=/path/to/libtorch/

It's expected that the SuperCollider repo is cloned at `../supercollider` relative to this repo. If
it's not: add the option `-DSC_PATH=/path/to/sc/source`.

    cmake .. -DSC_PATH=/path/to/sc/source

You may want to manually specify the install location in the first step to point it at your
SuperCollider extensions directory: add the option `-DCMAKE_INSTALL_PREFIX=/path/to/extensions`.
Note that you can retrieve the Extension path from sclang with `Platform.userExtensionDir`

    cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/extensions

To enable platform-specific optimizations:

    cmake .. -DNATIVE=ON

Finally, use CMake to build the project:

    cmake --build . --config Release
    cmake --build . --config Release --target install


### Developing

The usual `regenerate` command was disabled because `CmakeLists.txt` needed to be manually edited to include libtorch.

#### Design

Torchscripts are loaded on scsynth asynchronously via a PlugIn cmd, and stored with an index, similar to bufnums. Since loaded models can have settable attributes, it is supported to load the same torchscript file more than once, resulting in multiple independent instances of the same model. 

Once a model is loaded, informations about which methods and attributes it offers are cached and optionally communicated to sclang. In lack of a better way to send a complex reply to the client, scsynth will write model informations to a yaml file, which the client can then read.


