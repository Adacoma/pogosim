# Change Log
All notable changes to this project will be documented in this file.

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

