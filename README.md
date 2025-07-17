# Pogosim ![Badge CI](https://github.com/Adacoma/pogosim/actions/workflows/ci.yaml/badge.svg) ![Version](https://img.shields.io/badge/version-v0.10.6-blue)
Pogosim is a simulator for the [Pogobot robots](https://pogobot.github.io/). It aims to reproduce the C API used on the robots, so that the exact same code can be used in simulations as in robotic experiments.

Pogosim is coded in C++20 and C17, using SDL2 and Box2D 3.0.

## Overview
Here are the simulated runs of several examples (C code found [here](examples)).
![gallery](https://github.com/Adacoma/pogosim/blob/main/.description/gallery.gif)


## Install on Linux
To install it on *Debian/Ubuntu* (tested: 24.04 LTS), use the following commands. The process will be similar on other Linux distributions.

First, install the necessary packages:
```shell
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential cmake git libboost-system-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-gfx-dev libsdl2-ttf-dev \
    libyaml-cpp-dev libspdlog-dev libfmt-dev \
    wget unzip ca-certificates lsb-release

# Install Apache Arrow
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update
sudo apt install -y -V libarrow-dev
```

Then compile and install Box2D 3.0:
```shell
git clone https://github.com/erincatto/box2d.git
cd box2d
git checkout 28adacf82377d4113f2ed00586141463244b9d10
mkdir build && cd build
cmake -DBOX2D_BUILD_DOCS=OFF -DGLFW_BUILD_WAYLAND=OFF -DCMAKE_INSTALL_PREFIX=/usr  ..
cmake --build .
sudo make install
cd ../..
```

Clone the pogosim repository, compile pogosim and install it:
```shell
git clone https://github.com/Adacoma/pogosim.git
cd pogosim
./build.sh 
```

**REMEMBER TO ALWAYS RECOMPILE with "./build.sh" after downloading a new version of Pogosim!**


## Install on WSL
Just follow the previous section to install on Ubuntu 24.04+ using WSL.
If the simulator is really slow, it may be due to a bug with SDL2 on WSL, as explained [here](https://github.com/libsdl-org/SDL/issues/6333#issuecomment-1293872149) and [here]( https://github.com/lem-project/lem/issues/1332). If this is the case, just add:
```shell
export SDL_RENDER_DRIVER=software
```
before launching the simulator.

**REMEMBER TO ALWAYS RECOMPILE with "./build.sh" after downloading a new version of Pogosim!**


## Install on MacOSX
The installation requires brew to be installed on your computer, cf brew documentation [here](https://brew.sh/).

You can then install the necessary packages to compile Pogosim:
```shell
brew install cmake boost sdl2 sdl2_image sdl2_gfx sdl2_ttf yaml-cpp spdlog apache-arrow pkg-config fmt
```

Then compile and install Box2D 3.0:
```shell
git clone https://github.com/erincatto/box2d.git
cd box2d
git checkout 28adacf82377d4113f2ed00586141463244b9d10
mkdir build && cd build
cmake -DBOX2D_BUILD_DOCS=OFF -DGLFW_BUILD_WAYLAND=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_INSTALL_LIBDIR=/usr/local/lib -DCMAKE_INSTALL_INCLUDEDIR=/usr/local/include ..
cmake --build .
sudo make install
cd ../..
```

Clone the pogosim repository, compile pogosim and install it:
```shell
git clone https://github.com/Adacoma/pogosim.git
cd pogosim
./build.sh 
```

**REMEMBER TO ALWAYS RECOMPILE with "./build.sh" after downloading a new version of Pogosim!**


## Quickstart

### Launch example codes
Example codes are compiled every time you launch the "./build.sh" script, alongside the rest of the Pogosim code.

To launch examples code you can use the following commands:
```shell
./examples/helloworld/helloworld -c conf/simple.yaml              # Hello world, just robots rotating left then right. The first robot prints "HELLO WORLD !" messages
./examples/run_and_tumble/run_and_tumble -c conf/simple.yaml      # A very simple implementation of the run-and-tumble algorithm for locomotion
./examples/hanabi/hanabi -c conf/simple.yaml                      # A simple code to showcase the diffusion of information in a swarm. Immobile robots by default (uncomment "MOVING_ROBOTS" to make then move)
./examples/phototaxis/phototaxis -c conf/phototaxis.yaml          # An example showcasing phototaxis, with a fixed light spot in the middle of the arena
./examples/walls/walls -c conf/walls_and_membranes.yaml           # An multi-controller example where robots can identify the presence of fixed walls (through Pogowalls) or mobile walls (through membranes).
./examples/ssr/ssr -c conf/ssr.yaml         # More complex example. "Simple" implementation of the SSR algorithm from https://arxiv.org/abs/2403.17147  You can test it for a disk and annulus arena (see conf/ssr.yaml to change the arena).
./examples/coverage_neighbors_novelty/coverage_neighbors_novelty -c conf/coverage_neighbors_novelty.yaml   # More complex run-and-tumble example, with two objectives: neighbor novelty, and isolation avoidance (as a proxy to global coverage)
./examples/IMU/IMU -c conf/simple.yaml                            # A run-and-tumble example showing how to retrieve IMU information (gyroscope, accelerometer, temperature sensor)
```


### Simple way to create a new pogobot/pogosim project
Just copy the directory "template\_prj":
```shell
cp -R template_prj ~/my_new_pogo_prj
```

*Option 1*: Create simlinks to the necessary libraries:
```shell
cd ~/my_new_pogo_prj
ln -s PATH/TO/pogosim       # https://github.com/Adacoma/pogosim
ln -s PATH/TO/pogobot-sdk   # https://github.com/nekonaute/pogobot-sdk
ln -s PATH/TO/pogo-utils    # If you use pogo-utils in your project. https://github.com/Adacoma/pogo-utils
```

*Option 2*: Set environment variables to link to necessary libraries:
Add the following lines in the configuration file of your shell (e.g. ~/.bashrc for BASH or ~/.zshrc for ZSH):
```shell
export POGO_SDK=/ABSOLUTE/PATH/TO/pogobot-sdk
export POGOSIM_INCLUDE_DIR=/ABSOLUTE/PATH/TO/pogosim/src
export POGOUTILS_INCLUDE_DIR=/ABSOLUTE/PATH/TO/pogo-utils/src
```

*Option 3*: Edit the Makefile so that the following variables contain the paths to the necessary libraries:
```shell
POGO_SDK?=PATH/TO/pogobot-sdk
POGOSIM_INCLUDE_DIR?=PATH/TO/pogosim/src
POGOUTILS_INCLUDE_DIR?=PATH/TO/pogo-utils/src
```


After using any of these 3 options, you can compile the project:
```shell
make clean sim  # To compile the simulation
# OR
make clean bin  # To compile the binary for real Pogobots
# OR
make clean all  # To compile both the simulation and Pogobot binaries
```

By default, the name of the created simulation binary corresponds to the name of the parent directory of the project. You can then launch it using:
```shell
make clean sim
./template_prj -c conf/test.yaml        # If the parent directory is "template_prj"
```


### Controlling the GUI
Here is a list of shortcuts that can be used to control the GUI:
 - F1: Help message
 - F3: Slow down the simulation
 - F4: Speed up the simulation
 - F5: Show/Hide the communication channels
 - F6: Show/Hide the lateral LEDs
 - F7: Show/Hide the light level
 - ESC: quit the simulation
 - SPACE: pause the simulation
 - DOWN, UP, LEFT, RIGHT: move the visualisation coordinates
 - Right-Click + Mouse move: move the visualisation coordinates
 - PLUS, MINUS or Mouse Wheel: Zoom up or down
 - 0: Reset the zoom and visualization coordinates


### Compile a binary for the real Pogobots
Download the [pogobot-SDK](https://github.com/nekonaute/pogobot-sdk) somewhere:
```shell
git clone https://github.com/nekonaute/pogobot-sdk.git
```

Edit "~/my\_pogobot\_project/Makefile" to set the path of the pogobot-sdk: change the value of variable "POGO\_SDK".

Use the following commands to compile the binary:
```shell
cd ~/my_pogobot_project
make clean && make bin
```

The binary should be compiled correctly, and you can then use the usual commands to upload it to a robot. E.g. through:
```shell
make connect TTY=/dev/ttyUSB0
```
Inside the robot prompt, type "enter" to obtain a new prompt line. 
If you connect to the robot through a Progboard, you can use the command "serialboot" to upload the code. Cf the [pogobot-SDK documentation](https://github.com/nekonaute/pogobot-sdk) for more details.
If you use the IR remote device, follow the instructions described [here](https://github.com/nekonaute/pogobot/blob/main/readme-irRemote.md).

#### Compiling binaries for multi-categories projects
Note that if the project involve several robot categories that each have a different code (e.g. example "./examples/walls" where Pogobots, Pogowalls and Membranes have different code), it is possible to specify the category you want to compile for, using the following command:
```shell
cd ~/my_pogobot_project
make clean && make bin ROBOT_CATEGORY=robots  # where "ROBOT_CATEGORY" is the category specified in the pogobot_start function.
```
Note that "robots" is the default category (e.g. with pogobot\_start calls with only 2 arguments).

For instance, the following command can be used to compile the example code "./examples/walls" for Pogowalls:
```shell
cd examples/walls
make clean && make bin ROBOT_CATEGORY=walls
```


### Headless mode
To launch your simulation in headless mode (while still exporting png files of the traces), use the "-g" command line parameter. E.g.:
```shell
./my_pogobot_project -c conf/test.yaml -g
```
The simulator is far faster in headless mode than in windowed mode.


### Command line parameters of the simulator
```shell
Usage: pogosim [options]
Options:
  -c, --config <file>             Specify the configuration file.
  -g, --no-GUI                    Disable GUI mode.
  -v, --verbose                   Enable verbose mode.
  -nr, --do-not-show-robot-msg    Suppress robot messages.
  -P, --progress                  Show progress output.
  -V, --version                   Show version information.
  -h, --help                      Display this help message.
```
- Parameter "-c" must always be provided, and corresponds to the YAML configuration file to use. See "conf/test.yaml" for an example.
- Parameter "-g" enables headless mode: no GUI shown, but the program still export frames.
- Parameter "-v" enables verbose mode (show debug messages).
- Parameter "-nr" disables messages from the robots (printf in robot code).
- Parameter "-P" displays a progress bar of the simulation, depending on the parameter value "simulation\_time" defined in the configuration file.


## Troubleshooting

### In headless/Pogobatch mode, I get an SDL-related error
If you get this error:
```
INFO: Failed to initialize SDL: offscreen not available  
Error: Error while initializing SDL
```
It means that you compiled Pogosim with an SDL version < 2.0.22. Headless mode is not available in this version of SDL.
To have access to a newer version, you can:
    - update your system.
    - use apptainer/singularity (cf related section below) to create an Ubuntu 24.04 image with a newer version of SDL.


## Access the pose and states of the robots in Python
After a simulation is executed, it can periodically store the pose (position and orientation) and internal states of each robot into a data file.
This feature can be enabled in the configuration file, with entries:
```yaml
enable_data_logging: true                   # Set to true to enable the generation of a data file
data_filename: "frames/data.feather"        # Path of the generated data file
save_data_period: 1.0       # In s          # Save data every 1.0 second
```

The data is stored as an Apache Arrow Feather file, a standard and convenient format to store large dataframes.
As such, it can easily be imported in Python by using Pandas:
```yaml
import pandas as pd
df = pd.read_feather("frames/data.feather")
print(df)

       time robot_category  robot_id  pogobot_ticks         x         y       angle  
0      1.00          walls     65535             63  5.001000  5.001000    0.000000  
1      1.00      membranes     65534             63  6.824879  4.867349         NaN  
2      1.00         robots         0             63  4.038734  0.959281    1.953128  
3      1.00         robots         1             63  1.023770  8.115510    1.303922  
4      1.00         robots         2             63  1.965905  3.455247   -2.005039  
...     ...            ...       ...            ...       ...       ...         ...  
1423  14.13         robots        95            884  0.654096  9.729611   -0.386542  
1424  14.13         robots        96            884  8.011082  6.353422    1.812981  
1425  14.13         robots        97            884  4.807075  7.854455    1.177420  
1426  14.13         robots        98            884  4.021065  9.730083    1.626796  
1427  14.13         robots        99            883  9.031286  1.299344   -0.168437  

[1428 rows x 7 columns]
```

Custom columns can be added into this file by using the callback mechanism. See examples "hanabi" (simple) and "ssr" (complex) for more information.


## Launch several runs in Parallel, with different configuration options
We provide Python scripts that can launch several runs of Pogosim in parallel, and compile the results from all runs into a single dataframe.

To install it, use the following command:
```shell
pip install pogosim
```

Or, just you want to compile it yourself:
```shell
cd scripts
./setup.py sdist bdist_wheel
pip install -U .
cd ..
```

Afterwards, you can use the pogobatch script to launch several runs of simulation in parallel (or in a cluster), with a given configuration:
```shell
pogobatch -c conf/test.yaml -S ./examples/hanabi/hanabi -r 10 -t tmp -o results
```
This command with launch 10 runs of the Hanabi example using configuration file conf/test.yaml. Temporary files of the runs will be stored in the "tmp" directory.
After all runs are completed, the script will compile a dataframe of all results and save it into "results/result.feather". It can then be opened as described in previous section. An additional column "run" is added to the dataframe to distinguish results from the different runs.

It is also possible to launch the pogobatch script on several variations of a given configuration, e.g. with a list of different numbers of robots or arena. The list of possibly configuration combination is specified in the configuration file, by adding a subkey "batch\_options" with the list of possible values.
E.g.:
```yaml
arena_file:        # Test the results on two arenas
    batch_options: ["arenas/disk.csv", "arenas/arena8.csv"]
    default_option: arenas/disk.csv    # OPTIONAL: Value to use for "arena_file" when this configuration is used directly by the simulator, not pogobatch
objects:
    robots:
        type: pogobot       # Category type pertaining to Pogobots
        nb:                 # Number of objects (Pogobots) in this category
            batch_options: [100, 200]          # Test the results on three different swarm sizes
            default_option: arenas/disk.csv    # OPTIONAL: Value to use for "objects.robots.nb" when this configuration is used directly by the simulator, not pogobatch
        geometry: disk                  # Pogobots are always disk-shaped
        radius: 26.5                    # In mm


# Format of the generated dataframes, one for each configuration
result_filename_format: "result_{objects.robots.nb}.feather"

# List of new columns to add in the generated dataframes
result_new_columns: ["arena_file"]
```
These configuration entries specify that either 100 or 200 robots should be considered, on arenas "disk" and "8", resulting in 4 possibly configurations. The configuration entry "result\_filename\_format" corresponds to the name of a given configuration combination.
See "conf/batch/test.yaml" for a complete example. The entry "result\_new\_columns" indicates which columns (and associated configurations) are *stored* inside feather files as additional columns.

You can use pogobatch script on this compounded configuration file to launch several runs on each configuration combination:
```shell
pogobatch -c conf/batch/test.yaml -S ./examples/hanabi/hanabi -r 10 -t tmp -o results

Found 6 combination(s) to run.
Task: Config file /home/syemn/data/prj/pogosim/tmp/combo_kdmvxpzf.yaml -> Output: results/result_50.feather
Task: Config file /home/syemn/data/prj/pogosim/tmp/combo_cqfk3lrl.yaml -> Output: results/result_100.feather
Task: Config file /home/syemn/data/prj/pogosim/tmp/combo_ckx7t160.yaml -> Output: results/result_150.feather
Task: Config file /home/syemn/data/prj/pogosim/tmp/combo_401kcmam.yaml -> Output: results/result_50.feather
Task: Config file /home/syemn/data/prj/pogosim/tmp/combo_wiilbu4e.yaml -> Output: results/result_100.feather
Task: Config file /home/syemn/data/prj/pogosim/tmp/combo_q3yxls9z.yaml -> Output: results/result_150.feather
Removed stale result file: results/result_50.feather
Removed stale result file: results/result_150.feather
Removed stale result file: results/result_100.feather
Launch → tmp tmp/run_c03834a11a87406384efdcf8d2376dd4.feather  (will merge into results/result_50.feather)
Combined data saved to tmp/run_c03834a11a87406384efdcf8d2376dd4.feather
Created results/result_50.feather with 49500 rows
Launch → tmp tmp/run_2628efb844fc4e2e83da56f7e98e8084.feather  (will merge into results/result_100.feather)
Combined data saved to tmp/run_2628efb844fc4e2e83da56f7e98e8084.feather
Created results/result_100.feather with 99000 rows
Launch → tmp tmp/run_3c868b57dff0483c920ebf656b8c2eff.feather  (will merge into results/result_150.feather)
Combined data saved to tmp/run_3c868b57dff0483c920ebf656b8c2eff.feather
Created results/result_150.feather with 148500 rows
Launch → tmp tmp/run_9e26f0240f8a4105aaf6851e5614ab93.feather  (will merge into results/result_50.feather)
Combined data saved to tmp/run_9e26f0240f8a4105aaf6851e5614ab93.feather
Appended 49500 rows to results/result_50.feather
Launch → tmp tmp/run_143db3464fdd456c8cac379f621ae474.feather  (will merge into results/result_100.feather)
Combined data saved to tmp/run_143db3464fdd456c8cac379f621ae474.feather
Appended 99000 rows to results/result_100.feather
Launch → tmp tmp/run_84b9da00ad624d788d7bcce6c301f8ca.feather  (will merge into results/result_150.feather)
Combined data saved to tmp/run_84b9da00ad624d788d7bcce6c301f8ca.feather
Appended 148500 rows to results/result_150.feather
Batch run completed. Generated output files:
 - results/result_50.feather
 - results/result_100.feather
 - results/result_150.feather
 - results/result_50.feather
 - results/result_100.feather
 - results/result_150.feather
```

If you want to implement more complex deployment behaviors, you can write your own Python scripts and extend the class "pogosim.pogobatch.PogobotBatchRunner".


## Install and use the simulator in an Apptainer/Singularity container
The main image definition file for apptainer is based on Ubuntu 24.04 LTS ("pogosim-apptainer.def"). An alternative image based on Ubuntu 22.04 LTS can also be found ("pogosim-apptainer\_ubuntu22.04.def").

To build the image:
```shell
sudo apptainer build --sandbox -F pogosim.simg pogosim-apptainer.def
```
Or, if you want to use Clang instead of GCC:
```shell
sudo apptainer build --sandbox -F --build-arg USE_CLANG=true pogosim.simg pogosim-apptainer.def
```


Use the image to compile a pogosim project:
```shell
cd ~/my_pogobot_project
apptainer exec /PATH/TO/pogosim.simg make clean sim
```
Note that your current directory should be a subpath of your home (~) directory -- elsewise apptainer/singularity cannot access it by default.

Then the simulator can be launched with:
```shell
apptainer exec /PATH/TO/pogosim.simg ./my_pogobot_project -c conf/test.yaml
```



## Generate gif files of the traces
By default, the frames of a simulated run are stored in the directory "frames/" (cf variable "frames\_name" in the configuration file).
They can be assembled into an animated gif file using various commands, such as mencoder, ffmpeg, or ImageMagick.
We recommend the program [gifski](https://gif.ski/), a very high-quality GIF encoder:
```shell
gifski -r 20 --output animation.gif frames/*png
```


## Development

If you want to compile the pogosim library with debugging symbols and options (e.g. -Og -g compilation parameters), you can specify the configuration Debug to the build script:
```shell
./build.sh Debug
```

This is useful if you want to debug an error, e.g. with:
```shell
gdb --args ./examples/run_and_tumble/run_and_tumble -c conf/simple.yaml 
```

To generate Doxygen documentation:
```shell
doxygen
```
you can then open "html/index.html".
To compile the latex report:
```shell
cd latex
make
```
This will generate a PDF report named "latex/refman.pdf".

## Authors

 * Leo Cazenille: Main author and maintainer.
    * email: leo "dot" cazenille "at" gmail "dot" com
 * Nicolas Bredeche
    * email: nicolas "dot" bredeche "at" sorbonne-universite "dot" fr


## Citing

```bibtex
@misc{pogosim,
    title = {pogosim: A simulator for Pogobot robots},
    author = {Cazenille, L., Bredeche, N.},
    year = {2025},
    publisher = {Github},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/Adacoma/pogosim}},
}
```


