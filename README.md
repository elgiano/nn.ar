# `NN.ar()`

[nn_tilde](https://github.com/acids-ircam/nn_tilde) adaptation for SuperCollider: load torchscripts for real-time audio processing.

### Description
It has most features of nn_tilde:
- interface for any available model method (e.g. forward, encode, decode)
- interface for setting model attributes (and a debug interface to print their values when setting them)
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

// 3. attributes:
NN(\msprior).attributes;
// -> [ listen, temperature, learn_context, reset ]

// attributes can be set only per-UGen, using the 'attributes' argument
{
    var in, latent, modLatent, prior, resynth;

    in = SoundIn.ar();
    latent = NN(\rave, \encode).ar(2048, in);
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(MouseY.kr.exprange(0.1, 30)).range(-0.5, 0.5)
    };
    prior = NN(\msprior, \forward).ar(2048, latent
        // attributes are set when their value changes
        attributes: [
            // here we use latch to limit the setting rate to once per second
            temperature: Latch.kr(LFPar.kr(0.1).range(0, 2), LFPulse.kr(1))
        ],
        debug: 1 // print attribute values when setting them
    );
    resynth = NN(\rave, \decode).ar(2048, prior.drop(-1);

    resynth
}.play;

```

### Installation

#### Download a pre-built release

- Download the latest release for your OS on the [Releases page](https://github.com/elgiano/nn.ar/releases).
- Extract the archive and copy the `nn.ar` folder to your SuperCollider Extensions folder

**Note for mac users**: binaries are not signed, so you need to run the following in SuperCollider to bypass macos security complaints:
```supercollider
runInTerminal("xattr -d -r com.apple.quarantine" + shellQuote(Platform.userExtensionDir +/+ "nn.ar"))
```
Failing to do so can produce errors like:
```
"libc10.dylib" is damaged and can’t be opened. You should move it to the Bin.
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

**Buffering and external threads**
Most nn operation, from loading to processing, are resource intensive and can block the DSP chain. In order to alleviate this, but costing extra latency, we adopted the same buffering method as nn_tilde. When buffering is enabled (by default if not on an NRT server), model loading, processing and parameter setting are done asynchronously on an external thread.
The only issue still present with this approach is that we have to wait for the thread to finish before we can destroy the UGen. This currently blocks the DSP chain when the UGen is destroyed.

**Model and description loading**
For processing purposes, models are loaded by NNUGen. This is because each processing UGen needs a separate instance of the model, since multiple inferences on the same model are not guaranteed not to interfere with each other. So now models are loaded and destroyed with the respective UGen, similarly to what happens in MaxMSP and PureData. However, since we couldn't find in SuperCollider a convenient method to send messages to single UGens, we opted for loading model descriptions separately, so that paths and attribute names could be referenced as integer indexes.

1. `NN.load` loads the model on scsynth and save its description in a global store, via a PlugIn cmd. Once a model is loaded, informations about which methods and attributes it offers are cached and optionally communicated to sclang. In lack of a better way to send a complex reply to the client, scsynth will write model informations to a yaml file, which the client can then read.
2. When creating an UGen, a model, its method and attribute names are referenced by their integer index 
3. The UGen then loads its own independent instance of the model, at construction time, or in an external thread if buffering is enabled
4. When the UGen is destroyed, its model is unloaded as well.

**Attributes**
Since each UGen has its own independent instance of a model, attribute setting is only supported at the UGen level. Currently, attributes are updated each time their value changes, and we suggest to use systems like `Latch` to limit the setting rate (see example above).


#### Latency considerations (RAVE)

RAVE models can exhibit an important latency, from various sources. Here is what I found:

**tl;dr**: if latency is important, consider training with --config causal.

- first obvious source of latency is buffering: we fill a bufferSize of data before passing it to the model. With most of my rave v2 models, this is 2048/44100 = ~46ms.
- then processing latency: on my 2048/44100 rave v2 models, on my i5 machine from 2016, this is between 15 and 30ms. That is very often bigger than 1024/44100 (~24ms, my usual hardware block size), so I have to use the external thread all the time to avoid pops.
- rave intrinsic latency: cached convolutions introduce delay. From the paper [Streamable Neural Audio Synthesis With Non-Causal Convolutions](https://arxiv.org/abs/2204.07064), this can be about 650ms, making up for most of the latency on my system. Consider using models trained with `--config causal`, which reduces this latency to about 5ms, at the cost of a "small but consistent loss of accuracy".
- transferring inputs to an external thread doesn't contribute significantly to latency (I've measured delays in the order of 0.1ms)
