# Change Log
All notable changes to this project will be documented in this file.

## [0.10.8] - 2025-11-23

### Added
 - pogobatch: allow sequential backend + option to enable GUI
 - add script+conf to benchmark speed of Pogosim
 - add implementation of the "pogobot\_infrared\_get\_receiver\_error\_counter" function of the Pogobot API
 - add a new "am.py" script to compute Active-Matter statistics of robot dynamics. Split results by arena
 - add example 'avoid\_walls' that show robots avoiding active walls
 - add default active wall user code functions in pogosim.{h,c} -- active wall default behavior can just be set using one line (cf example "avoid\_walls")
 - PogobotObject: add support for systematic angular biases
 - add support for periodic boundary conditions ("empty.csv" or "torus.csv" configuration files, or "boundary\_condition: periodic" in config). find neighbors respects periodic boundary conditions. Disable pogowall creation on periodic BC
 - add Vicsek example, with heading estimation and wall avoidance from pogo-utils. Use cluster-level U-turns to avoid walls.
 - add Toner-tu example, extending the Vicsek example with crowding/pressure terms + others.
 - network script: create violin comparing arena eigenvalues
 - examples using pogo-utils will be peacefully skipped if pogo-utils is not found (Makefile)
 - add the global\_step callback, executed once per step
 - configuration: allow dotted paths
 - README: link to arxiv paper
 - pogobatch: allow hierarchical options + dataframe columns
 - configuration: allow hierarchical options format from pogobatch (always take the 'default' option). Hierarchical options can now be named.
 - pogoptim: add hierarchical categorical optimization domains
 - support for float16 in data logs (2 bytes). Takes far less space than double (8 bytes)
 - add a configuration file containing noisy locomotion

### Changed
 - intangible objects pose is now always saved as NaN
 - run\_and\_tumble example: randomly select run or tumble at init
 - photostart: always starts at light value = 0, before photo start light spike happens
 - locomotion.py script now splits results by arena
 - CI: now compile and launch/test template\_prj alongside the baseline examples
 - update apptainer instructions in the README, avoid the --sandbox parameter

### Fixed
 - major fix: when receiving from all directions, create 4 differents msg reception calls, rather than one call with \_receiver\_ir\_index=ir\_all. This aligns with the Pogolib.
 - major fix: remove boost\_system from the Makefile, as it is not compatible with boost 1.89
 - avoid crash if no robot are created, put warning instead
 - network script: fix crash when input dataframes did not have 'run' and 'arena\_file' columns
 - fix pogosim.arenas script import from python
 - pogobatch/pogoptim: fix logging - debug messages were not filtered correctly when pogoptim called pogobatch routines
 - fix CMakeLists to avoid returning a warning on some computers when compiling examples


## [0.10.7] - 2025-09-11

### Added
 - README: add a mandatory export for WSL users
 - add example 'moving\_oscillators' (Kuramono-style)
 - add example 'push\_sum' (canonical gossip algorithm)
 - add example 'active\_brownian\_particles' that implements ABP + wall avoidance
 - add example 'lighthouse\_localization' (robots detect their X,Y position using 2 beams of light)
 - add a new object: rotating ray of light (lighthouse style, useful for localization), and object 'alternating\_rays\_of\_light' for lighthouse localization
 - possible to set the number of light map bins in the configuration files
 - colormaps: add an hsv\_to\_rgb function
 - neighbors.py: add kNN stats, degree histograms, fano wrt time
 - parameter 'photosensors\_systematic\_bias\_domain' to simulate real-robot photosensor biases
 - parameter 'photosensors\_noise\_stddev' to simulate gaussian noise over the detected light levels
 - add 'arenas.py' and 'coverage.py' (voronoi plots). Add mean CV across arenas computation and plots
 - CI: add WSL2 test build + launch hanabi to test if headless mode works in CI + test pogobatch
 - locomotion.py: add MSD computation
 - add the pogoptim script, used to optimize parameters of a Pogosim simulation, with 3 optimizers: random search. CMA-ES (from pyCMA), MAP-Elites (from QDpy)
 - add example configurations for pogoptim
 - GUI: new options to show/hide current time and the scale bar

### Changed
 - geometry: reduce min gap between objects in random formation
 - changed GUI window title to Pogosim + version
 - reduce simulation\_time in conf/mini.yaml and conf/pair.yaml to allow super fast computation by default

### Fixed
 - MAJOR FIX: fix bug in Makefiles: GCC could sometimes delete used code (e.g. communication routines) if FLTO was not enabled
 - fix photosensors: return different values depending on the sensor\_number instead of a single value
 - fix photosensors position: 180, 320, 40 degrees
 - fix colormaps to use the [0,25] domain instead of the [0,255] domain
 - update requirements.txt of pogosim python scripts
 - fix physical objects: only compute acceleration/velocity stats if the object allows it (e.g. not membranes)
 - script utils.py: fix metadata import from feather, configuration is now always a dict (never a string)
 - pogobatch: correctly save original configuration in file dataframe metadata
 - Disable some free/destroy/quit functions, as they can crash with older versions of SDL2
 - fix Boost-related CMake errors on MacOSX
 - CI: remove brew install cmake, to avoid fatal errors in recent github MacOSX images


## [0.10.6] - 2025-07-17

### Added
 - add example 'coverage\_neighbors\_novelty' (including python scripts), a complex run-and-tumble example with 2 objectives: isolation prevention (proxy for coverage) and neighbors novelty
 - add 'random\_near\_walls' and 'aligned\_random\_near\_walls' formations
 - neighbor\_counter example: export the list of neighbors
 - allow initialization of char* parameters from configuration files
 - implement ICM-20689 IMU statistics and pogolib API (gyroscope, accelerometer, temperature sensor)
 - add temperature information in all config files
 - add example 'IMU' that showcases how to retrieve IMU information (gyroscope, accelerometer, temperature sensor)

### Changed
 - objects and robots: disable initial velocity

### Fixed
 - fix bug in 'random' initial formation where points could be situated directly on the walls
 - fix "data\_set\_value\_string" to correctly take a char const* as input
 - configuration: fix 'exists' method, that returned true even if a configuration entry did not exist


## [0.10.5] - 2025-07-02

### Added
 - Add support for conditional data export activation: in the callback "callback\_export\_data", a data row is only exported if activated
 - Experimental support for C++20 Pogobot user programs (without RTTI, and heapless)
 - Add example 'neighbor\_counter', that counts the number of neighbors of the robots, allowing for a given duration for message exchanges
 - Add example 'run\_tumble\_wait', a more complex version of the run&tumble algorithm than the previous "run\_and\_tumble" example
 - Add chessboard and aligned\_chessboard initial formation: robots are placed in a grid with constant distance between two points

### Changed
 - Colormaps are now included directly when importing pogobase.h, no need to include another .h file

### Fixed
 - Arena 'arrow2': update pdf and svg to match the csv
 - No longer possible to set motor levels (through the 'pogobot\_motor\_power\_set' function) above motorFull
 - Untangible objects are no longer assembled in the default formation
 - Fix neighbors computation: radius was not used correctly
 - Fix bug in examples Makefile that prevented make bin to compile/link correctly
 - Fix bug in neighbors computation: hashcells size were too small: max\_comm\_radius -> (max\_comm\_radius + max\_robot\_radius)


## [0.10.4] - 2025-05-19

### Added
 - Support for light gradients (cf phototaxis example) in "static\_light object" 
 - In configs: "save\_data\_period" can be set to negative values to disable it
 - Add new Python script to create locomotion-related plots
 - README: add troubleshooting section
 - Add config 'example\_imported\_formation.yaml'

### Changed
 - GUI: camera movement using right click depends on zoom level

### Fixed
 - Major fix in pogosim main loop: message recv did not call "pogobot\_infrared\_clear\_message\_queue" by default. Messages were never deleted in the queue.
 - Fix X,Y exports in CSV/feather files: store values in mm instead of pixels
 - Dynamic message success rate computation: "msg\_size" now takes into account start+end bytes and 2 CRC16
 - Minor fixes in several Python scripts



## [0.10.3] - 2025-05-09

### Added
 - arena geometry (vectors of points of the walls) are now exported as metadata of the feather files
 - add python script that scales csv arenas to a given total surface
 - add a python script to compute the optimal p\_send to use given cluster size and msg size
 - add support for the 'imported' initial formation, importing x,y,angle poses from a csv/feather file

### Changed
 - refactoring of several C++ code files: parts of render.{h,cpp} were moved to geometry.{h,cpp}; parts of simulation.{h,cpp} were moved to main.{h,cpp}
 - updated information about pogobatch in README
 - feather export: check if all fields are specified when adding row, check if there are no duplicate fields/metadata

### Fixed
 - fix message sending when using ir\_all and \*omni\* functions
 - the "star" arena is now regular and normalized
 - fix bug where walls and untangible objects were taken into account to create the initial formation
 - fix bug where SDL>2.0.5 generated two repeated events when keys are pressed



## [0.10.2] - 2025-05-05

### Added
 - support for occlusions in detecting neighbors
 - communication: "show\_communication\_channels\_above\_all" parameter (+ F5 in GUI) to show communication channels above/below all other objects
 - communication channels now show bidirectionnal links using arrows
 - pogobatch: support retries if the simulation segfaults
 - initial formations: add 'lloyd' and 'power\_lloyd' algorithms for uniform distribution, and 'aligned\_random' formation (same behavior as "random" in v0.10.1)
 - add macros to specify Pogosim version numbers, inspired from SDL
 - save configuration in feather files, as YAML-formatted metadata

### Changed
 - only draw robot arrow if it is large enough with current zoom
 - configuration: automatically convert float from/to int if needed
 - random orientation in initial formation
 - power\_lloyd is now the default formation
 - changed algorithm to simulate communication channels: enable occlusions by default

### Fixed
 - fix a bug where robots don't update correctly their neighbor list when moving
 - fix headless mode to support SDL2 versions below 2.0.22



## [0.10.1] - 2025-05-01

### Added
 - the "init\_from\_configuration" macro can be used to initialize any globals from a configuration file "parameters.\*" entry
 - pogobatch now supports the config parameter 'result\_new\_columns' that automatically add columns from batch options
 - normal simulations now accept pogobatch config files. Default options (instead of batch\_options) are provided with configuration key 'default\_option'
 - add config params to specify max robot speed
 - support for SDL headless mode if GUI is disabled
 - new parameters for random formation: formation\_attempts\_per\_point and formation\_max\_restarts, to specify how many attempts should be tried to find initial positions
 - add "formation\_min\_space\_between\_neighbors" and "formation\_max\_space\_between\_neighbors" parameters in config
 - implementation of "pogobot\_motor\_dir_\*" and "pogobot\_motor\_power_\*" functions from the Pogolib
 - run\_and\_tumble example: parameter to allow backward run phases

### Changed
 - remove locomotion noise in most configs

### Fixed
 - add pyarrow as requirement of pogosim python scripts
 - msg reception now updates "\_receiver\_ir\_index"



## [0.10.0] - 2025-04-23

### Added
 - Support for object creation, specified in the configuration file
 - Support for objects/robots categories, with different user code for each category
 - Support for light levels in different parts of the simulation, through "static\_light" objects
 - Support for passive\_object, Pogowalls, Pogobjects and Membranes
 - Temporal noise, to simulate lags in robots
 - Dynamic message success rate, using a formula fitted on experimental communication profiles (fitting Python script in the "scripts" direction)
 - Support for the 4 IR emitter in Pogobots, instead of just one
 - Option in config + GUI to (1) show/hide communication channels between robots (F5), show/hide lateral LEDs (F6) and show/hide light level (F7)
 - Use fmt library instead of C++20 std::format, to ensure compat with older versions of GCC/Clang
 - Add README entry about WSL install
 - Add README entry about compiling multi-category projects
 - Add new examples: phototaxis, walls
 - Global float arrays can now be set from a configuration file with function init\_float\_array\_from\_configuration
 - Check category of current robot in user code, via the functions get\_current\_robot\_category() and current\_robot\_category\_is(category)
 - Added a gallery of examples in the README

### Changed
 - Configuration files now have a different structure, to allow object creation and specification
 - No linear/angular locomotion noise by default: "robot\_linear\_noise\_stddev" and "robot\_angular\_noise\_stddev" are both 0.0 by default
 - Improve robot rendering
 - Communication\_radius in configuration file is now from IR emitter to robot border
 - Callback\_create\_data\_schema is now global, not linked to each robot
 - Changed Pogobatch configuration format to handle array-based configuration values
 - Update Hanabi example: photo start based in light difference rather than direct threshold

### Fixed
 - Fixed initial\_formation=disk (previously, objects where not assembled in a disk-shaped pattern)
 - The run\_and\_tumble example now compiles with make clean bin (previously, attempted to compile simulation-related code for robot binaries)
 - Fix arena creation bug with duplicate vertexes
 - Fix bug in exporting schema and callback 'callback\_export\_data': did not update current robot



## [0.9.0] - 2025-03-02

### Added
 - Support for MacOS X
 - Pogobatch script to easily launch several runs of simulation
 - Move visualisation coordinates by left/right/top/bottom keys: depends on current zoom
 - Register parameter values from configuration file
 - Add quiet mode
 - README: tutorial explaining how to launch example code and load data files with pandas

### Changed

### Fixed
 - Large number of minor fixes so that the code can compile on MacOS X with Apple Clang

