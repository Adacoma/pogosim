# Pogosim ![Badge CI](https://github.com/Adacoma/pogosim/actions/workflows/ci.yaml/badge.svg) ![Version](https://img.shields.io/badge/version-v0.10.10-blue)
Pogosim is a simulator for the [Pogobot robots](https://pogobot.github.io/). It aims to reproduce the C API used on the robots, so that the exact same code can be used in simulations as in robotic experiments. An extensive description of the Pogosim can be found in this [article](https://arxiv.org/pdf/2509.10968) or [here (RG link)](https://www.researchgate.net/publication/395526571_Pogosim_--_a_Simulator_for_Pogobot_robots). The full Doxygen documentation of Pogosim can be found [here](https://adacoma.github.io/pogosim/).

Pogosim is coded in C++20 and C17, using SDL2 and Box2D 3.x.

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

Then compile and install Box2D 3.x:
```shell
git clone https://github.com/erincatto/box2d.git
cd box2d
git checkout 28adacf82377d4113f2ed00586141463244b9d10
mkdir build && cd build
cmake -DBOX2D_SAMPLES=OFF -DBOX2D_UNIT_TESTS=OFF -DBOX2D_DOCS=OFF -DCMAKE_INSTALL_PREFIX=/usr ..
cmake --build .
sudo cmake --install .
cd ../..
```

Clone the pogosim repository, compile pogosim and install it:
```shell
git clone https://github.com/Adacoma/pogosim.git
cd pogosim
./build.sh 
```

**REMEMBER TO ALWAYS RECOMPILE with "./build.sh" after downloading a new version of Pogosim!**

Note that some (optional) advanced examples need the [pogo-utils](https://github.com/Adacoma/pogo-utils) library -- see below section "Simple way to create a new pogobot/pogosim project" to know how to register associated environment variables so that the Makefiles can assess pogo-utils. If pogo-utils is not present, those examples won't be compiled, but the rest of Pogosim will compile without errors.


## Install on WSL
Just follow the previous section to install on Ubuntu 24.04+ using WSL.
If the simulator is really slow, it may be due to a bug with SDL2 on WSL, as explained [here](https://github.com/libsdl-org/SDL/issues/6333#issuecomment-1293872149) and [here]( https://github.com/lem-project/lem/issues/1332). If this is the case, just add:
```shell
export SDL_RENDER_DRIVER=software
export LIBGL_ALWAYS_SOFTWARE=1
```
before launching the simulator.

**REMEMBER TO ALWAYS RECOMPILE with "./build.sh" after downloading a new version of Pogosim!**

Note that some (optional) advanced examples need the [pogo-utils](https://github.com/Adacoma/pogo-utils) library -- see below section "Simple way to create a new pogobot/pogosim project" to know how to register associated environment variables so that the Makefiles can assess pogo-utils. If pogo-utils is not present, those examples won't be compiled, but the rest of Pogosim will compile without errors.


## Install on MacOSX
The installation requires brew to be installed on your computer, cf brew documentation [here](https://brew.sh/).

You can then install the necessary packages to compile Pogosim:
```shell
brew install cmake boost sdl2 sdl2_image sdl2_gfx sdl2_ttf yaml-cpp spdlog apache-arrow pkg-config fmt
```

Then compile and install Box2D 3.x:
```shell
git clone https://github.com/erincatto/box2d.git
cd box2d
git checkout 28adacf82377d4113f2ed00586141463244b9d10
mkdir build && cd build
cmake \
    -DBOX2D_SAMPLES=OFF \
    -DBOX2D_UNIT_TESTS=OFF \
    -DBOX2D_DOCS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_INSTALL_LIBDIR=/usr/local/lib \
    -DCMAKE_INSTALL_INCLUDEDIR=/usr/local/include \
    ..
cmake --build .
sudo cmake --install .
cd ../..
```

Clone the pogosim repository, compile pogosim and install it:
```shell
git clone https://github.com/Adacoma/pogosim.git
cd pogosim
./build.sh 
```

**REMEMBER TO ALWAYS RECOMPILE with "./build.sh" after downloading a new version of Pogosim!**

Note that some (optional) advanced examples need the [pogo-utils](https://github.com/Adacoma/pogo-utils) library -- see below section "Simple way to create a new pogobot/pogosim project" to know how to register associated environment variables so that the Makefiles can assess pogo-utils. If pogo-utils is not present, those examples won't be compiled, but the rest of Pogosim will compile without errors.


## Quickstart

### Launch example codes
Example codes are compiled every time you launch the "./build.sh" script, alongside the rest of the Pogosim code.

To launch examples code you can use the following commands:
```shell
./examples/helloworld/helloworld -c conf/simple.yaml                    # Hello world, just robots rotating left then right. The first robot prints "HELLO WORLD !" messages.
./examples/run_and_tumble/run_and_tumble -c conf/simple.yaml            # A very simple implementation of the run-and-tumble algorithm for locomotion.
./examples/blooming/blooming -c conf/blooming.yaml                      # A simple code to showcase the diffusion of information in a swarm. LED colors correspond to the hop distance to a random robot seed (white LED). Immobile robots by default (set 'moving_robots').
./examples/phototaxis/phototaxis -c conf/phototaxis.yaml                # An example showcasing phototaxis, with a fixed light spot in the middle of the arena
./examples/phototaxis_gradient//phototaxis_gradient -c conf/phototaxis_gradient.yaml   # Phototaxis with a gradient of light: the robots are always searching for the most lighted spot by following light levels gradients
./examples/walls/walls -c conf/walls_and_membranes.yaml                 # An multi-controller example where robots can identify the presence of fixed walls (through Pogowalls) or mobile walls (through membranes).
./examples/avoid_walls/avoid_walls -c conf/active_objects.yaml          # Show a simple pogowalls-avoidance system, and active objects (that can emit messages like pogowalls)
./examples/ssr/ssr -c conf/ssr.yaml                                     # More complex example. "Simple" implementation of the SSR algorithm from https://arxiv.org/abs/2403.17147  You can test it for a disk and annulus arena (see conf/ssr.yaml to change the arena).
./examples/coverage_neighbors_novelty/coverage_neighbors_novelty -c conf/coverage_neighbors_novelty.yaml   # More complex run-and-tumble example, with two objectives: neighbor novelty, and isolation avoidance (as a proxy to global coverage)
./examples/IMU/IMU -c conf/simple.yaml                                  # A run-and-tumble example showing how to retrieve IMU information (gyroscope, accelerometer, temperature sensor)
./examples/push_sum/push_sum -c conf/simple.yaml                        # Canonical example of the push-sum gossip algorithm.
./examples/moving_oscillators/moving_oscillators -c conf/simple.yaml    # Showcases a Kuramoto-style moving oscillators swarm achieving synchronization. The robots move according to a run-and-tumble algorithm.
./examples/lighthouse_localization/lighthouse_localization -c conf/lighthouse.yaml  # Robot estimate their X,Y position using two rotating lighthouse inspired by the **Valve SteamVR** tracking system, often used with drone localization.
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
The GUI is composed of several parts, as presented in this figure:
![GUI](https://github.com/Adacoma/pogosim/blob/dev/.description/gui.png)

Here is a list of shortcuts that can be used to control the GUI:
 - F1: Help message
 - F2: Show/Hide the trajectory traces
 - F3: Slow down the simulation
 - F4: Speed up the simulation
 - F5: Show/Hide the communication channels, below/above the other objects
 - F6: Show/Hide the lateral LEDs
 - F7: Show/Hide the light level
 - F8: Show/Hide the current time and scale bar
 - ESC: quit the simulation
 - SPACE: pause the simulation
 - S: pause and advance the simulation by one tick
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
  -q, --quiet                     Enable quiet mode (ouput only warning and errors on terminal).
  -nr, --do-not-show-robot-msg    Suppress robot messages.
  -s, --seed <int>                Seed the simulator RNG.
  -P, --progress                  Show progress output.
  -V, --version                   Show version information.
  -h, --help                      Display this help message.
```
- Parameter "-c" must always be provided, and corresponds to the YAML configuration file to use. See "conf/test.yaml" for an example.
- Parameter "-s" is the random seed -- can also be specified in YAML configuration files, with the one provided in the CLI be prioritized.
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

### Specifying logged fields

By default, all built-in fields and all user-defined fields created in the data schema callback are logged. The output can be restricted from the configuration:
```yaml
# Optional: only write these columns to the data file.
# Remove a field from this list to stop logging it.
data_logger_fields:
  - time
  - robot_category
  - robot_id
  - pogobot_ticks
  - x
#   - y     # a built-in field <- not logged, as it is commented
  - angle
  - energy  # e.g. a custom field <- logged
#   - fitness # e.g. a custom field <- not logged in this case

# Optional: only log rows for these robot categories.
data_logger_category:
  - robots
  - prey
```

Useful when a user program defines many custom fields, but a specific experiment only needs some of them in the output file. Moreover, depending on the experiment this can significantly reduce the size of the logged files. Similarly, `data_logger_category` lets an experiment follow only selected object or robot categories.

### Data storage - retrieve simulation dataframes and configuration parameters

The data is stored as an Apache Arrow Feather file, a standard and convenient format to store large dataframes.
As such, it can easily be imported in Python by using Pandas:
```python
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

Or alternatively, using the "pogosim" python package (cf next section for installation procedure):
```python
import pogosim.utils as pu
df, meta = pu.load_dataframe("frames/data.feather")
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

# "meta" (metadata) contains a dict with the program version, the arena polygons and the YAML configuration used.
# To print out the configuration dictionnary:
import pprint
pprint.pprint(meta['configuration'])

{'GUI': True,
 'GUI_speed_up': 10.0,
 'arena_file': 'arenas/disk.csv',
 'arena_surface': '1.0e6',
 'arena_temperature': 25.0,
 'boundary_condition': 'solid',
 'chessboard_distance_between_neighbors': 110,
 'communication_ignore_occlusions': False,
 'console_filename': 'frames/console.txt',
 'data_filename': 'frames/data.feather',
 'delete_old_files': True,
 'enable_console_logging': True,
 'enable_data_logging': True,
 'formation_cluster_at_center': True,
 'frames_name': 'frames/f{:010.4f}.png',
 'initial_formation': 'random',
 'initial_formation_root_object': 'arena',
 'log_format': 'default',
 'mm_to_pixels': 0.531604,
 'objects': {'global_light': {'geometry': 'global',
                              'light_mode': 'static',
                              'photo_start_at': 1.0,
                              'photo_start_duration': 1.0,
                              'photo_start_value': 32767,
                              'type': 'static_light',
                              'value': 200},
 [...]
}
```

Custom columns can be added into this file by using the callback mechanism. See examples "blooming" (simple) and "ssr" (complex) for more information.



## Install and use the simulator in an Apptainer/Singularity container
The main image definition file for apptainer is based on Ubuntu 24.04 LTS ("pogosim-apptainer.def"). An alternative image based on Ubuntu 22.04 LTS can also be found ("pogosim-apptainer\_ubuntu22.04.def").

To simply retrieve a pre-built apptainer image of the Pogosim main-branch v0.10.10, use the following command:
```shell
apptainer pull library://leo.cazenille/pogosim/pogosim-full:v0.10.10
```
If apptainer has an old version (1.4.x) and complains about a missing library client, ensure you run the following commands:
```shell
apptainer remote add --no-login SylabsCloud cloud.sycloud.io
apptainer remote use SylabsCloud
```
then re-run the pull command.

Alternatively, to build the image on your computer:
```shell
sudo apptainer build -F pogosim.sif pogosim-apptainer.def
```
Or, if you want to use Clang instead of GCC:
```shell
sudo apptainer build -F --build-arg USE_CLANG=true pogosim.sif pogosim-apptainer.def
```


Use the image to compile a pogosim project:
```shell
cd ~/my_pogobot_project
apptainer exec /PATH/TO/pogosim.sif make clean sim
```
Note that your current directory should be a subpath of your home (~) directory -- elsewise apptainer/singularity cannot access it by default.

Then the simulator can be launched with:
```shell
apptainer exec /PATH/TO/pogosim.sif ./my_pogobot_project -c conf/test.yaml
```


## Launch several runs in Parallel, with different configuration options
We provide the "Pogobatch" tool with Pogosim. It allows you to launch parallel Pogosim tasks locally or on clusters, using the rundra batch meta-scheduler. 
A complete guide can here found [here](https://github.com/Adacoma/pogosim/blob/dev/docs/pogobatch-rundra-guide.md).



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


## Instructions for AI agents: reproducible headless simulations and data collection

For parameter sweeps, cluster execution, or quantitative analysis, run Pogosim
without the GUI, pass an explicit seed, and log only the fields required by the
analysis. A reproducible simulation is determined by the Pogosim source/build,
the effective YAML configuration, and the CLI seed.

### Analysis-oriented configuration

Starting from any valid Pogosim configuration for your controller, review the
following simulator-generic settings:

```yaml
# Geometry and population
boundary_condition: solid
arena_file: arenas/disk.csv
arena_surface: 1.0e6       # mm^2
initial_formation: random

objects:
  robots:
    nb: 100
    # Keep the remaining geometry, dynamics, sensing, and communication
    # properties required by the selected controller/configuration.

# Time
simulation_time: 120.0     # requested simulated duration, seconds
time_step: 0.01            # physics step, seconds

# Headless execution and output
GUI: false
save_video_period: -1.0    # disable PNG frame export
enable_console_logging: false

enable_data_logging: true
data_filename: frames/data.feather
save_data_period: 1.0      # requested logging period, seconds

# Restricting the schema substantially reduces output size in large sweeps.
data_logger_fields:
  - time
  - robot_category
  - robot_id
  - x
  - y

# Restrict rows when the simulation contains categories that are not part of
# the analysis.
data_logger_category:
  - robots
```

`arena_file` is resolved in the simulator's execution environment. Prefer a
repository-relative path for normal local runs. In a container or cluster
workflow, use a path known to exist inside the runtime, such as a file in the
staged source tree.

### Launch with an explicit seed

```bash
./examples/PROGRAM/PROGRAM \
  --config conf/experiment.yaml \
  --seed 17 \
  --no-GUI \
  --quiet \
  --do-not-show-robot-msg
```

The short equivalents are `-c`, `-s`, `-g`, `-q`, and `-nr`. The command-line
seed takes precedence over a `seed` value in the YAML file. Record the exact
seed for every stochastic run; rerunning with the same seed also requires the
same program build and effective configuration.

Headless mode disables interactive rendering, but data and explicitly enabled
frame outputs are still written. Set `save_video_period: -1.0` when images are
not needed.

### Data format and metadata

Pogosim writes Apache Arrow Feather files. They can be read directly with
Pandas:

```python
import pandas as pd

data = pd.read_feather("frames/data.feather")
```

Use PyArrow when schema metadata is also needed:

```python
import pyarrow.feather as feather
import yaml

table = feather.read_table("frames/data.feather")
data = table.to_pandas()
metadata = {
    key.decode("utf-8"): value.decode("utf-8")
    for key, value in (table.schema.metadata or {}).items()
}

# Pogosim stores configuration and arena geometry as YAML text.
configuration_text = metadata.get("configuration")
configuration = (
    yaml.safe_load(configuration_text)
    if configuration_text is not None
    else None
)
arena_polygons = (
    yaml.safe_load(metadata["arena_polygons"])
    if "arena_polygons" in metadata
    else None
)
```

The built-in pose fields use the following units:

| Field | Meaning | Unit |
|---|---|---|
| `time` | simulated time | seconds |
| `robot_id` | identifier within a robot category | integer |
| `x`, `y` | center position | millimetres |
| `angle` | orientation | radians |
| `pogobot_ticks` | controller tick counter | ticks |

Custom fields registered by a controller may use controller-specific units.
Document those units beside the experiment configuration.

### Timing expectations

`simulation_time` and `save_data_period` are requested values. Actual logged
timestamps are produced on simulator ticks, so the last timestamp need not be
exactly equal to `simulation_time`, and adjacent logged times can differ
slightly from the requested period. Analysis should use the `time` column
rather than reconstructing timestamps from row numbers.

Before combining repeated runs, verify that they contain compatible time grids,
robot counts, categories, and schemas. For time-series statistics, use each
robot's first available observation as its baseline unless the experiment
defines another reference time.

### Boundary conditions, arena size, and population density

The physical interpretation of displacement depends on the boundary condition:

- `solid` confines robots. Long-time mean squared displacement and related
  measures eventually approach a finite-arena plateau.
- periodic or topology-changing boundaries can create coordinate discontinuities.
  Displacements must be unwrapped according to that boundary model before a
  conventional unbounded-space MSD is computed.

Changing `objects.<category>.nb` while keeping `arena_surface` fixed changes
population density, collision frequency, and often communication connectivity.
When comparing population sizes, decide explicitly whether arena area or
density should remain constant and record that choice.

### Repeated runs and parameter sweeps

Use Pogobatch when an experiment needs multiple seeds, multiple configuration
choices, or merged Feather results. Pogobatch records the effective choice and
seed for each atomic run, and its merger adds stable `run`, `seed`, and
`retry_attempt` columns to the combined dataset. For remote execution, Pogobatch
can use Rundra to prepare the simulator, submit scheduler work, retrieve raw
outputs, and merge the task shards.


## Authors

 * Leo Cazenille: Main author and maintainer.
    * email: leo "dot" cazenille "at" gmail "dot" com
 * Nicolas Bredeche
    * email: nicolas "dot" bredeche "at" sorbonne-universite "dot" fr


## Citing

```bibtex
@article{cazenille2025pogosim,
  title={Pogosim-a Simulator for Pogobot robots},
  author={Cazenille, Leo and Macabre, Loona and Bredeche, Nicolas},
  journal={arXiv preprint arXiv:2509.10968},
  year={2025}
}
```

