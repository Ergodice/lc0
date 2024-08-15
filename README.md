[![CircleCI](https://circleci.com/gh/LeelaChessZero/lc0.svg?style=shield)](https://circleci.com/gh/LeelaChessZero/lc0)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/3245b83otdee7oj7?svg=true)](https://ci.appveyor.com/project/leelachesszero/lc0)

# Lc0

Lc0 is a UCI-compliant chess engine designed to play chess via neural network, specifically those of the [LeelaChessZero project](https://lczero.org).
This is an experimental repository for testing new features to the Lc0 chess engine. Updates are made frequently so there may be bugs. Please report any issues you find to the Lc0 Discord. All elo measurements were calculated on A100 with unbalanced human openings.
Many of these features are taken from the Katago engine. A detailed description of these methods can be found [here](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md). A list of the improvements follows.

## Quickstart

Create a folder for the engine and download one of the binaries [here](https://ci.appveyor.com/project/Ergodice/lc0/builds/49100387).
Create a subfolder called "nets" and add a network (either BT3, T3, or BT4) from [here](https://storage.lczero.org/files/networks-contrib/).
Add a file titled lc0.config with the following parameters:

```
--weights=...
--backend-opts=policy_head=vanilla,value_head=winner
```

The weights argument should be the path to your network of choice.




## New features in this fork


### Fused multihead attention
Almaudoh and Ankan have optimized the attention calculation using cutlass, giving a 10% speedup to larger models.

To build with Cutlass, you'll have to download the Cutlass 2.11 branch. This isn't the latest, and newer versions may break things!

`git clone -b 2.11 https://github.com/NVIDIA/cutlass.git`

and set the `CUTLASS_INCLUDE_PATH` in `build.cmd` to the include directory. Also set `CUTLASS` to true.


### Desperation

When a position seems to be very drawish/winnish, search can be widened by increasing CPUCT. 
If the absolute value of the position's q value is at most `desperation-low` or at least
`desperation-high` the CPUCT valued is increased by a factor of `desperation-multiplier`. 
The prior weight is the weight at which the effect is a half of its max value. This setting is NOT RECOMMENDED, 
but the following settings can be used for testing.

```
--desperation-multiplier=1.5
--desperation-low=0.3
--desperation-high=0.7
--desperation-prior-weight=500
--use-desperation=true 
```

### Policy boosting

Some positions like

```
r3kb1N/pp2n1p1/2q1b3/5pPp/3pp2N/P2n3B/1P3P1P/1RBQ2KR w - - 5 11
r3kb1r/1b3ppp/p2p1n2/3Pn3/Pq1N4/1B6/1P3PPP/R1BQR1K1 w kq - 1 3
```

with resp. best moves `Bg2` and `h3` have a strong move practically ignored because it has low policy.
This feature sets a lowest baseline policy for the top few moves so that this doesn't happen.
The top `top-policy-num-boost` are treated as if they had policy at least `top-policy-boost`. 
There is also a "tier two" for moves that are still ranked highly but not in the top few. Removing regressed by a couple elo at LTC.

Recommended settings below.
Enabled by default.

```
--top-policy-num-boost=3
--top-policy-boost=0.05
--use-policy-boosting=true
```


### CPUCT exponent

The exponent in the CPUCT formula can now be configured. Default is
```
--cpuct-exponent=0.5
```


### Node reporting

The nps statistic used to represent the number of playouts per second, which may not be what the user wants. The new `--reported-nodes` option specifies what the node count and nps statistic represent. 
The default is `nodes` for LowNodes. The other options are `queries` for neural network queries and `playouts` or `legacy` for playouts (what we used to use).


### 50 move rule caching

The first 64 plies out of 100 are partitioned into 8 equally sized buckets. Before a position is queried for NN evaluation, the 50 move rule ply is checked. If the bucket containing the position already has nodes, the eval is copied from the one with the most visits.
The speedup can be anywhere from 5% to 50% depending on how transposition-heavy the position is. The gain was measured at 20 elo on STC. The nodes per second statistic is now calculated by the number of true nodes (called LowNode in the code) rather than edges (technically playouts, called Node in the code) so the reported value may be lower than on previous dag versions.

This feature is enabled by default and can be disabled by specifying `--move-rule-bucketing=false` in the config.

### Multiple output heads

The BT3 generation of nets now has multiple value and policy heads that you can choose among. You always want the `winner` head for value. There are two policy heads, `vanilla` and `optimistic`.
The vanilla head is similar to what we used before, and the optimistic head upweights moves that the net believes may be exceptionally strong, an idea taken from the Katago engine. The optimistic head is recommended as it adds roughly 20 elo at LTC. The Katago team observed that this improvement is similar to increasing the net size by roughly 40%.

The head you choose is specified in the backend options. For example, if you have a single GPU and want to use the optimistic policy head and winner value head, then you can use this can easily be specified as `--backend-opts=policy_head=vanilla,value_head=winner`.

If you have are using more than one GPU then you have to set up the backend for each GPU. An example is
`--backend-opts=(gpu0,policy_head=optimistic,value_head=winner),(gpu1,policy_head=optimistic,value_head=winner)`.

### Uncertainty Weighting

Identical to the [Katago implementation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#uncertainty-weighted-mcts-playouts), but the error uses only the winrate. The new parameters are

```
--uncertainty-weighting-cap
--uncertainty-weighting-coefficient
--uncertainty-weighting-exponent
--use-uncertainty-weighting
```


Enabled by default. With BT3, the gain was 20 elo at LTC with

```
--uncertainty-weighting-cap=1.03
--uncertainty-weighting-coefficient=0.13
--uncertainty-weighting-exponent=-1.76
--use-uncertainty-weighting=true
```

### CPUCT Uncertainty

Katago [found that](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#dynamic-variance-scaled-cpuct) scaling CPUCT to match the variance of rewards gains elo.
We didn't find elo from that approach but did find that varying CPUCT based on the uncertainty of the position improved performance. The new parameters are

```
--use-cpuct-uncertainty=true
--cpuct-uncertainty-max-uncertainty=0.347
--cpuct-uncertainty-min-uncertainty=0.0
--cpuct-uncertainty-min-factor=0.87
--cpuct-uncertainty-max-factor=1.78
```

This was found to gain [8 elo at LTC](https://bench.plutie.ca/test/134/). 


### Correction history
Identical to the [Katago implementation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#subtree-value-bias-correction). To form our buckets, we use the position of all pawns and kings, the number of each piece type, the last moved piece and its pick-up and put-down squares, and what piece was captured iwth that move if it was a capture.

The new parameters are

```
--use-correction-history=true
--correction-history-lambda=0.3
```

The gain was [8 elo at LTC](https://bench.plutie.ca/test/70/).

### CPUCT Utility Variance Scaling

THIS HAS BEEN REPLACED BY A CPUCT UNCERTAINTY FEATURE, DO NOT USE.

Identical to the [Katago implementation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#dynamic-variance-scaled-cpuct). The new parameters are

```
--cpuct-utility-stdev-prior 
--cpuct-utility-stdev-scale
--cpuct-utility-stdev-prior-weight
--use-variance-scaling
```

and the tuned values for masterkni's T1 are

```
--cpuct=2.3097
--fpu-value=0.5619
--cpuct-utility-stdev-prior=0.2289
--cpuct-utility-stdev-scale=0.3437
--cpuct-utility-stdev-prior-weight=10.0
--wdl-calibration-elo=3400
--use-variance-scaling=true
```

You must set `--use-variance-scaling=true` to turn it on. For now the setting is NOT RECOMMENDED.



## Downloading source

Lc0 can be acquired either via a git clone or an archive download from GitHub. Be aware that there is a required submodule which isn't included in source archives.

For essentially all purposes, including selfplay game generation and match play, we highly recommend using the latest `release/version` branch (for example `release/0.29`), which is equivalent to using the latest version tag.

Versioning follows the Semantic Versioning guidelines, with major, minor and patch sections. The training server enforces game quality using the versions output by the client and engine.


Download using git:

```shell
git clone -b release/0.29 --recurse-submodules https://github.com/LeelaChessZero/lc0.git
```

If you have cloned already an old version, fetch, view and checkout a new branch:
```shell
git fetch --all
git branch --all
git checkout -t remotes/origin/release/0.29
```


If you prefer to download an archive, you need to also download and place the submodule:
 * Download the [.zip](https://api.github.com/repos/LeelaChessZero/lc0/zipball/release/0.29) file ([.tar.gz](https://api.github.com/repos/LeelaChessZero/lc0/tarball/release/0.29) archive is also available)
 * Extract
 * Download https://github.com/LeelaChessZero/lczero-common/archive/master.zip (also available as [.tar.gz](https://github.com/LeelaChessZero/lczero-common/archive/master.tar.gz))
 * Move the second archive into the first archive's `libs/lczero-common/` folder and extract
 * The final form should look like `<TOP>/libs/lczero-common/proto/`

Having successfully acquired Lc0 via either of these methods, proceed to the build section below and follow the instructions for your OS.


## Building and running Lc0

Building should be easier now than it was in the past. Please report any problems you have.

Aside from the git submodule, lc0 requires the Meson build system and at least one backend library for evaluating the neural network, as well as the required `zlib`. (`gtest` is optionally used for the test suite.) If your system already has this library installed, they will be used; otherwise Meson will generate its own copy of the two (a "subproject"), which in turn requires that git is installed (yes, separately from cloning the actual lc0 repository). Meson also requires python and Ninja.

Backend support includes (in theory) any CBLAS-compatible library for CPU usage, such as OpenBLAS or Intel's DNNL or MKL. For GPUs, OpenCL and CUDA+cudnn are supported, while DX-12 can be used in Windows 10 with latest drivers.

Finally, lc0 requires a compiler supporting C++17. Minimal versions seem to be g++ v8.0, clang v5.0 (with C++17 stdlib) or Visual Studio 2017.

*Note* that cuda checks the compiler version and stops even with newer compilers, and to work around this we have added the `nvcc_ccbin` build option. This is more of an issue with new Linux versions, but you can get around it by using an earlier version of gcc just for cuda. As an example, adding `-Dnvcc_ccbin=g++-9` to the `build.sh` command line will use g++-9 with cuda instead of the system compiler.

Given those basics, the OS and backend specific instructions are below.

### Linux

#### Generic

1. Install backend:
    - If you want to use NVidia graphics cards Install [CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn).
    - If you want to use AMD graphics cards install OpenCL.
    - if you want OpenBLAS version Install OpenBLAS (`libopenblas-dev`).
2. Install ninja build (`ninja-build`), meson, and (optionally) gtest (`libgtest-dev`).
3. Go to `lc0/`
4. Run `./build.sh`
5. `lc0` will be in `lc0/build/release/` directory
6. Unzip a [neural network](https://lczero.org/play/networks/bestnets/) in the same directory as the binary.

If you want to build with a different compiler, pass the `CC` and `CXX` environment variables:

    CC=clang-6.0 CXX=clang++-6.0 ./build.sh

#### Note on installing CUDA on Ubuntu

Nvidia provides .deb packages. CUDA will be installed in `/usr/local/cuda-10.0` and requires 3GB of diskspace.
If your `/usr/local` partition doesn't have that much space left you can create a symbolic link before
doing the install; for example: `sudo ln -s /opt/cuda-10.0 /usr/local/cuda-10.0`

The instructions given on the nvidia website tell you to finish with `apt install cuda`. However, this
might not work (missing dependencies). In that case use `apt install cuda-10-0`. Afterwards you can
install the meta package `cuda` which will cause an automatic upgrade to a newer version when that
comes available (assuming you use `Installer Type deb (network)`, if you'd want that (just cuda-10-0 will
stay at version 10). If you don't know what to do, only install cuda-10-0.

cuDNN exists of two packages, the Runtime Library and the Developer Library (both a .deb package).

Before you can download the latter you need to create a (free) "developer" account with nvidia for
which at least a legit email address is required (their website says: The e-mail address is not made public
and will only be used if you wish to receive a new password or wish to receive certain news or notifications
by e-mail.). Further they ask for a name, date of birth (not visible later on), country, organisation ("LeelaZero"
if you have none), primary industry segment ("Other"/none) and which development areas you are interested
in ("Deep Learning").


#### Ubuntu 18.04

For Ubuntu 18.04 you need the latest version of meson, libstdc++-8-dev, and clang-6.0 before performing the steps above:

    sudo apt-get install libstdc++-8-dev clang-6.0 ninja-build pkg-config
    pip3 install meson --user
    CC=clang-6.0 CXX=clang++-6.0 INSTALL_PREFIX=~/.local ./build.sh

Make sure that `~/.local/bin` is in your `PATH` environment variable. You can now type `lc0 --help` and start.

#### Ubuntu 16.04

For Ubuntu 16.04 you need the latest version of meson, ninja, clang-6.0, and libstdc++-8:

    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo apt-add-repository 'deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main'
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install clang-6.0 libstdc++-8-dev
    pip3 install meson ninja --user
    CC=clang-6.0 CXX=clang++-6.0 INSTALL_PREFIX=~/.local ./build.sh

Make sure that `~/.local/bin` is in your `PATH` environment variable. You can now type `lc0 --help` and start.

#### openSUSE (all versions)

Instructions, packages and tools for building on openSUSE are at [openSUSE_install.md](openSUSE_install.md)

#### Docker

Use https://github.com/vochicong/lc0-docker
to run latest releases of lc0 and the client inside a Docker container.


### Windows

Here are the brief instructions for CUDA/CuDNN, for details and other options see `windows-build.md`.

0. Install Microsoft Visual Studio (2017 or later)
1. Install [CUDA](https://developer.nvidia.com/cuda-zone)
2. Install [cuDNN](https://developer.nvidia.com/cudnn).
3. Install Python3
4. Install Meson: `pip3 install --upgrade meson`
5. Edit `build.cmd`:

* Set `CUDA_PATH` with your CUDA directory
* Set `CUDNN_PATH` with your cuDNN directory (may be the same with CUDA_PATH)

6. Run `build.cmd`. It will ask permission to delete the build directory, then generate MSVS project and pause.

Then either:

7. Hit `Enter` to build it.
8. Resulting binary will be `build/lc0.exe`

Or.

7. Open generated solution `build/lc0.sln` in Visual Studio and build yourself.

### Mac

First you need to install some required packages through Terminal:
1. Install brew as per the instructions at https://brew.sh/
2. Install python3: `brew install python3`
3. Install meson: `brew install meson`
4. Install ninja: `brew install ninja`
5. (For Mac OS 10.14 Mojave, or if the other step 5 fails):
 * Install developer tools: ``xcode-select --install``
 * When using Mojave install SDK headers: `installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /` (if this doesn't work, use `sudo installer` instead of just `installer`.)

Or.

5. (For MacOS 10.15 Catalina, or if the other step 5 fails): 
 * Install Xcode command-line tools: ``xcode-select --install``
 * Install "XCode Developer Tools" through the app store. (First one on the list of Apps if searched.)
 * Associate the SDK headers in XCode with a command: export CPATH=\`xcrun --show-sdk-path\`/usr/include
 
Now download the lc0 source, if you haven't already done so, following the instructions earlier in the page.

6. Go to the lc0 directory.
7. Run `./build.sh -Dgtest=false` (needs step 5)

### Raspberry Pi

You'll need to be running the latest Raspberry Pi OS "buster".

1. Install OpenBLAS

```shell
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS/
make
sudo make PREFIX=/usr install
cd ..
```

2. Install Meson

```shell
pip install meson
pip install ninja
```

3. Install compiler and standard libraries

```shell
sudo apt install clang-6.0 libstdc++-8-dev
```

4. Clone lc0 and compile

```shell
git clone https://github.com/LeelaChessZero/lc0.git
cd lc0
git submodule update --init --recursive
CC=clang-6.0 CXX=clang++-6.0 ./build.sh -Ddefault_library=static
```

5. The resulting binary will be in build/release

## Python bindings

Python bindings can be built and installed as follows.

```shell
pip install --user git+https://github.com/LeelaChessZero/lc0.git
```

This will build the package `lczero-bindings` and install it to your Python user install directory.
All the `lc0` functionality related to position evaluation is now available in the module `lczero.backends`.
An example interactive session can be found [here](https://github.com/LeelaChessZero/lc0/pull/1261#issuecomment-622951248).

## License

Leela Chess is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Leela Chess is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

### Additional permission under GNU GPL version 3 section 7

_The source files of Lc0 with the exception of the BLAS and OpenCL
backends (all files in the `blas` and `opencl` sub-directories) have
the following additional permission, as allowed under GNU GPL version 3
section 7:_

If you modify this Program, or any covered work, by linking or
combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
modified version of those libraries), containing parts covered by the
terms of the respective license agreement, the licensors of this
Program grant you additional permission to convey the resulting work.

