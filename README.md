# `NN.ar()`

[nn_tilde](https://github.com/acids-ircam/nn_tilde) adaptation for SuperCollider: load torchscripts for real-time audio processing.

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
  + [Pre-trained models](#pre-trained-models)
  + [Loading a model](#loading-a-model)
  + [Selecting a method](#selecting-a-method)
  + [Using attributes](#using-attributes)
  + [Buffer configuration](#buffer-configuration)
  + [Multichannel](#multichannel)
* [Building from source](#building-from-source)
* [Troubleshooting](#troubleshooting)
* [Design](#design)
  + [Latency considerations (RAVE)](#latency-considerations-rave)

## Features
It has most features of nn_tilde:
- interface for any available model method (e.g. forward, encode, decode)
- interface for setting model attributes (and a debug interface to print their values when setting them)
- processes real-time at different buffer sizes, on separate threads
- loads models asynchronously on scsynth

- tested so far only with [RAVE](https://github.com/acids-ircam/rave) (v1, v2, v3) and [msprior](https://github.com/caillonantoine/msprior) models
- tested so far only on CPU, on linux, windows and mac.

## Installation
### Download a pre-built release

- Download the latest release for your OS on the [Releases page](https://github.com/elgiano/nn.ar/releases/latest).
- Extract the archive and copy the `nn.ar` folder to your SuperCollider Extensions folder

**Note for mac users**: binaries are not signed, so you need to run the following in SuperCollider to bypass macos security complaints:
```supercollider
NN.deQuarantine;
s.reboot;
```
Failing to do so can produce errors like:
```
"libc10.dylib" is damaged and can’t be opened. You should move it to the Bin.
```

**Note for windows users**: you need to copy some DLLs to the folder where you have `scsynth.exe`.
1. Inside nn.ar folder you downloaded, there is a folder called `ignore`. Copy all the files inside this folder.
2. Find the folder where you have `scsynth.exe`, it's usually something like: `C:\Program Files\SuperCollider3.x.x\` (where 3.x.x is the version you have installed).
3. Paste all the .dll files you got in step 1 to the folder where `scsynth.exe` is.

Alternatively, you can use `NN.copyDLLs`, but you might have to run SuperCollider as Administrator:
```
s.quit; // server needs to be off for this to work
NN.copyDLLs
```

## Usage

### Pre-trained models
NN.ar is just an interface to load and play with pre-trained models, but doesn't include any. To begin with, you can find some published pre-trained RAVE models for real-time neural synthesis:
- from ACIDS-IRCAM: at <https://acids-ircam.github.io/rave_models_download>
- from iil (Intelligent Instruments Lab): at <https://huggingface.co/Intelligent-Instruments-Lab/rave-models>
For more information about RAVE, including instructions to train your own models, please visit [acids-ircam/rave](https://github.com/acids-ircam/rave)

### Loading a model
The following examples assumes you have a pre-trained model saved at `"~/Documents/Percussion.ts"`:
```supercollider
// Example:
// 1. load
s.waitForBoot {
    // when in a Routine, this method waits until the model has loaded
    NN.load(\ravePerc, "~/Documents/Percussion.ts");
    // when model has loaded, print a description of all methods and attributes
    NN(\ravePerc).describe;
}
```

### Selecting a method
Available methods are depending on the specific model you're using. RAVE models typically have three methods:
- forward: for audio-to-audio resynthesis
- encode: to encode live audio to latent values
- decode: to decode latent values to audio

```supercollider
// list all available methods of a pre-trained model
NN(\ravePerc).methods;
// -> NNModelMethod(forward: 1 ins, 1 outs), NNModelMethod(encode: 1 ins, 8 outs), ...

// 1. resynthesis using \forward method
{ NN(\ravePerc, \forward).ar(WhiteNoise.ar) }.play;

// 2. latent modulation using \encode and \decode
{
    var in, latent, modLatent, prior;
    // encode sound input to RAVE latent space
    latent = NN(\ravePerc, \encode).ar(SoundIn.ar);
    // add a random modulation to every dimension
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(MouseY.kr.exprange(0.1, 30)).range(-0.5, 0.5)
    };
    // resynthesize modulated latents
    NN(\ravePerc, \decode).ar(modLatent);
}.play;

// 3. manual latent navigation using only \decode
// here we assume ravePerc has 8 latent dimensions
Ndef(\rave) { NN(\ravePerc, \decode).ar(\latents.ar(0!8)) }.play;
// set all latents
Ndef(\rave).set(\latents, [0.75, 0.1, 1.5, 0.2, 0.3, 0.7, 1.2, 0.33])
// set any latent programmatically
Ndef(\rave).seti(\latents, 2, 0.5);
// simple GUI with sliders:
(
var win = View(bounds:800@200);
var sliders = 8.collect {|i| 
	Slider().background_(Color.rand)
	.action_{ |sl| Ndef(\rave).seti(\latents, i, sl.value) }
};
win.layout_(HLayout(*sliders)).front
)
// using the NodeProxyGui2 Quark, you automatically get the sliders you need:
// find it at https://github.com/madskjeldgaard/nodeproxygui2
Ndef(\rave).gui2

(
// maybe a more interesting multi-XY interface
var ev = EnvelopeView(bounds:500@500)
.drawLines_(false).drawRects_(true).style_(\dots)
.thumbSize_(48).background_(Color.clear)
.gridOn_(true).grid_(0.1@0.1).gridColor_(Color.grey(0.2))
.value_(Ndef(\rave).get(\latents).clump(2).flop)
.action_{|sl|
	Ndef(\rave).set(\latents, sl.value.flop.flatten)
}.front;
ev.value.shape[1].do {|v, i|
	ev.setFillColor(i, Color.rand);
	ev.setString(i, "z%,z%".format(i*2+1, i*2+2));
}
)
```

### Using attributes
Some models have custom attributes, to further condition their behavior. Which attribute and which behavior depends on the specific model. NN.ar supports setting and getting attribute values per-UGen, using the `attribute` argument:

```supercollider
// list model supported attributes
NN(\msprior).attributes;
// -> [ listen, temperature, learn_context, reset ]

// attributes can be set only per-UGen, using the 'attributes' argument
{
    var in, latent, modLatent, prior, resynth;

    in = SoundIn.ar();
    latent = NN(\rave, \encode).ar(in, 2048);
    modLatent = latent.collect { |l|
        l + LFNoise1.ar(MouseY.kr.exprange(0.1, 30)).range(-0.5, 0.5)
    };
    prior = NN(\msprior, \forward).ar(latent, 2048
        // attributes are set when their value changes
        attributes: [
            // here we use latch to limit the setting rate to once per second
            temperature: Latch.kr(LFPar.kr(0.1).range(0, 2), LFPulse.kr(1))
        ],
        debug: 1 // print attribute values when setting them
    );
    NN(\rave, \decode).ar(prior.drop(-1), 2048);
}.play;
```

### Buffer configuration
Like nn_tilde, nn.ar uses an internal circular buffer and runs neural network processing in a separate thread. The second argument of `NN(...).ar` controls this buffer's size, with 0 resulting in no buffering and no separate thread:

```supercollider
// large buffer
{ NN(\ravePerc, \forward).ar(WhiteNoise.ar, 8192) }.play;
// no buffer
{ NN(\ravePerc, \forward).ar(WhiteNoise.ar, 0) }.play;
```

### Multichannel
When supplying multiple inputs, NN.ar will process them using the same model, with batch processing.
```supercollider
// resynthesize three sine waves using three separate copies of the same model:
{ SinOsc.ar([100,200,300]).collect { |in| NN(\ravePerc, \forward).ar(in) } }.play

// with multi-channel: use a single model to resynthesize three inputs at the same time: 
{ NN(\ravePerc, \forward).ar(SinOsc.ar([100,200,300])) }.play

// example 2: pseudo-stereo: 
// - add 2 different noise vectors to the same latents
// - decode 2 slightly different channels, using a single model
{
    var latents = NN(\ravePerc, \encode).ar(SoundIn.ar);
    // latent size is 8
    var noiseVectors = 2.collect { LFNoise1.ar(1).range(-0.1, 0.1)}!latents.size }; 
    // noiseVectors shape is [2, 8]
    var modLatents = noiseVectors.collect { |noise| latents + noise };
    // modLatents shape is [2, 8]
    var out = NN(\ravePerc, \decode).ar(modLatents)
    // out shape is [2]
    out
}.play
```
Note that attributes don't cause multichannel expansion, precisely because we are loading a single model, that can only have one set of attributes.

## Building from source
If you compile SuperCollider from source, or if you want to enable optimizations specific to your machine, you need to build this extension yourself.

Build requirements:

- CMake >= 3.5
- [libtorch](https://pytorch.org/cppdocs/installing.html)
- SuperCollider source code: **at least version 3.13** (see the note below if you need to build with an earlier sc version)

If you don't have a copy of supercollider's source code, you can get one by:

    git clone --recursive https://github.com/supercollider/supercollider

### Clone the project:

    git clone https://github.com/elgiano/nn.ar
    cd nn.ar

### Download libtorch

We have built nn.ar releases with the following versions of libtorch:

- linux: [download v2.10.0 from pytorch.org](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip)
- macos arm64: [download v2.10.0 from pytorch.org](https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.10.0.zip).
- macos x64: [download v2.2.2 from pytorch.org](https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.2.2.zip).
- windows: [download v2.10.0 from pytorch.org](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.10.0%2Bcpu.zip)

Because of difficulties with torch CMake files and discrepancies in how different systems install libtorch, we don't recommend using system-wide installed libtorch. However, if you'd need to do it, you can enable finding and linking against system-installed torch with `-DSYSTEM_TORCH=ON`.

### Configure and build

    cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DTORCH_PATH=/path/to/libtorch/ \
        -DSC_PATH=/path/to/sc/source/ \
        -DCMAKE_INSTALL_PREFIX=/path/to/sc/extensions/ \
        -DNATIVE=ON

Options:
- **-DTORCH_PATH**: at the moment the recommended way of building nn.ar is to download a libtorch release (see above), unzip it and pass its path to cmake using `-DTORCH_PATH`. Systemwide installed torch will *not* be found automatically by default.
- **-DSC_PATH**: path to SuperCollider source.
- **-DCMAKE_INSTALL_PREFIX**: you may want to manually specify the install location to point directly at your
SuperCollider extensions directory. Note that you can retrieve the Extension path from sclang with `Platform.userExtensionDir`
- **-DNATIVE**: optional, enable platform-specific optimizations.

Finally, use CMake to build and install the project:

    cmake --build build --config Release --target install

> **Note: for building with any supercollider version earlier than 3.13**: nn.ar needs a macro called `ClearUnitOnMemFailed`, which was defined in supercollider starting from version 3.13. If for any reason you need to build nn.ar with a previous version of supercollider, you have to copy [these two macros](https://github.com/supercollider/supercollider/blob/a80436ac2cb22b8cef62192c86be2951639c184f/include/plugin_interface/SC_Unit.h#L83-L92) and put them in `NNModel.cpp`.

### Note for sc-plugin development

The usual `regenerate` command was disabled because `CMakeLists.txt` needed to be manually edited to include libtorch.

## Troubleshooting

### /cmd failed

> FAILURE IN SERVER /cmd failed

In SuperCollider 3.13 and earlier, this error message means that a plugin command is not found. In our case, it probably means that the server plugins failed to load. Check the output of `s.reboot` for errors, in most cases it is either because executables need to be de-quarantined on mac (see [Installation](#installation)) or because some libraries (e.g. libtorch) failed to load.

Later SuperCollider versions have a clearer error message:

> FAILURE IN SERVER /cmd /nn_load not found

### using NN.debug

If something doesn't work, you can get more informations posted to your Post window by activating debug mode:  

```supercollider
// server needs to be booted already
NN.debug(true);
```

This will print messages for a lot of backend operations when you run `NN.load()`.

For troubleshooting UGens, i.e. NN(...).ar, there is a debug option there too

```supercollider
{ NN(\myModel, \forward).ar(SoundIn.ar, debug: 2) }.play
```

## Design

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


### Latency considerations (RAVE)

RAVE models can exhibit an important latency, from various sources. Here is what I found:

**tl;dr**: if latency is important, consider training with --config causal.

- first obvious source of latency is buffering: we fill a bufferSize of data before passing it to the model. With most of my rave v2 models, this is 2048/44100 = ~46ms.
- then processing latency: on my 2048/44100 rave v2 models, on my i5 machine from 2016, this is between 15 and 30ms. That is very often bigger than 1024/44100 (~24ms, my usual hardware block size), so I have to use the external thread all the time to avoid pops.
- rave intrinsic latency: cached convolutions introduce delay. From the paper [Streamable Neural Audio Synthesis With Non-Causal Convolutions](https://arxiv.org/abs/2204.07064), this can be about 650ms, making up for most of the latency on my system. Consider using models trained with `--config causal`, which reduces this latency to about 5ms, at the cost of a "small but consistent loss of accuracy".
- transferring inputs to an external thread doesn't contribute significantly to latency (I've measured delays in the order of 0.1ms)
