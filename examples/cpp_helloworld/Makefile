# Get the absolute path of the Makefile
MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR := $(notdir $(patsubst %/,%,$(dir $(MAKEFILE_PATH))))

##### Detect OS #####
OS := $(shell uname)
ifeq ($(OS), Darwin)  # macOS
	WHOLE_ARCHIVE_FLAG = -Wl,-force_load
	SIM_LD_FLAGS_BOX2D = -L/usr/local/lib -Wl,-force_load,/usr/local/lib/libbox2d.a
	SIM_LD_FLAGS_POGOSIM = -Wl,-force_load,/usr/local/lib/libpogosim.a
	STDCPP_LIB = -lc++
	SIM_LD_EXT = -L/usr/local/lib -L/opt/homebrew/lib
	SIM_INCLUDES_EXT = -I/usr/local/include -I/opt/homebrew/include
else  # Linux
	SIM_LD_FLAGS_BOX2D = -lbox2d
	SIM_LD_FLAGS_POGOSIM = -Wl,--whole-archive -lpogosim -Wl,--no-whole-archive -lboost_system 
	WHOLE_ARCHIVE_FLAG = -Wl,--whole-archive
	STDCPP_LIB = -lstdc++
	SIM_LD_EXT = -L/usr/local/lib
	SIM_INCLUDES_EXT = -I/usr/local/include
endif

##### Settings for the pogobot binary #####
ROBOT_CATEGORY?=robots
BUILD_DIR?=./build
POGO_SDK?=../../pogobot-sdk
POGOSIM_INCLUDE_DIR?=../../src/
POGOSIM_SRC_DIR?=$(POGOSIM_INCLUDE_DIR)/pogosim/

POGO_SDK_TOOLS=$(POGO_SDK)/tools
POGO_SDK_INCS=$(POGO_SDK)/includes
POGO_SDK_LIBS=$(POGO_SDK)/libs

POGO_VAR=$(POGO_SDK_TOOLS)
ifneq ($(filter bin connect all default,$(MAKECMDGOALS)),)
include $(POGO_VAR)/variables.mak
include $(POGO_SDK_TOOLS)/common.mak
endif
ifeq ($(MAKECMDGOALS),)
include $(POGO_VAR)/variables.mak
include $(POGO_SDK_TOOLS)/common.mak
endif

TTY?=/dev/ttyUSB0

# Updated to handle both .c and .cpp files
C_SRCS := $(filter-out SDL_FontCache.c, $(notdir $(wildcard *.c)) $(notdir $(wildcard $(POGOSIM_SRC_DIR)/*.c)))
CPP_SRCS := $(notdir $(wildcard *.cpp))
ALL_SRCS := $(C_SRCS) $(CPP_SRCS)

C_OBJECTS=$(C_SRCS:.c=.o)
CPP_OBJECTS=$(CPP_SRCS:.cpp=.o)
ALL_OBJECTS=$(C_OBJECTS) $(CPP_OBJECTS)

C_OBJECTS_BUILD = $(patsubst %.c,build/bin/%.o,$(notdir $(C_SRCS)))
CPP_OBJECTS_BUILD = $(patsubst %.cpp,build/bin/%.o,$(notdir $(CPP_SRCS)))
OBJECTS_BUILD = $(C_OBJECTS_BUILD) $(CPP_OBJECTS_BUILD)

C_DEP_FILES=$(patsubst %.c,build/bin/%.d,$(notdir $(C_SRCS)))
CPP_DEP_FILES=$(patsubst %.cpp,build/bin/%.d,$(notdir $(CPP_SRCS)))
DEP_FILES=$(C_DEP_FILES) $(CPP_DEP_FILES)

##### Compiler settings for the simulator #####
SIM_CC = cc
SIM_CXX = c++
SIM_CFLAGS = -Wall -MMD -MP -O2 -std=c11 -I$(POGOSIM_INCLUDE_DIR) $(SIM_INCLUDES_EXT)
# Updated C++ flags for C++20 support
SIM_CXXFLAGS = -Wall -MMD -MP -O2 -std=c++20 -fno-exceptions -fno-rtti \
	$(shell pkg-config --cflags spdlog) -pthread -I$(POGOSIM_INCLUDE_DIR) $(SIM_INCLUDES_EXT)
SIM_LDFLAGS = $(SIM_LD_FLAGS_BOX2D)  $(SIM_LD_FLAGS_POGOSIM) -lyaml-cpp \
	$(shell pkg-config --libs spdlog) \
	-lSDL2 -lSDL2_gfx -lSDL2_ttf -lbox2d $(STDCPP_LIB) -lfmt -lm -larrow $(SIM_LD_EXT)

SIM_TARGET = $(MAKEFILE_DIR)

SIM_SRCS_CXX = $(wildcard *.cpp)
SIM_SRCS_C = $(wildcard *.c)
SIM_OBJECTS_CXX = $(patsubst %.cpp,build/sim/%.o,$(SIM_SRCS_CXX))
SIM_OBJECTS_C = $(patsubst %.c,build/sim/%.o,$(SIM_SRCS_C))
SIM_OBJECTS = $(SIM_OBJECTS_CXX) $(SIM_OBJECTS_C)

all: directories sim bin

# Ensure build directories exist
$(BUILD_DIR)/bin $(BUILD_DIR)/sim:
	mkdir -p $@

##### Rules for the simulator #####

$(BUILD_DIR)/sim/%.o: %.cpp | $(BUILD_DIR)/sim
	$(SIM_CXX) $(SIM_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/sim/%.o: %.c | $(BUILD_DIR)/sim
	$(SIM_CC) $(SIM_CFLAGS) -c $< -o $@

sim: directories $(SIM_TARGET)

$(SIM_TARGET): $(SIM_OBJECTS)
	$(SIM_CXX) $(SIM_CXXFLAGS) -o $@ $(SIM_OBJECTS) $(SIM_LDFLAGS)
	cp $(SIM_TARGET) $(BUILD_DIR)/sim/$(SIM_TARGET)

##### Rules for the Pogobot binary #####

bin: directories $(BUILD_DIR)/bin/firmware.bin

INCLUDES+=-I. -I$(POGO_SDK_INCS) -I$(POGOSIM_INCLUDE_DIR)

# Updated C++ flags for embedded systems - no exceptions, no RTTI, C++20 support
# Remove C-specific flags that cause warnings in C++
EMBEDDED_CXXFLAGS = -std=c++20 -fno-exceptions -fno-rtti -fno-use-cxa-atexit \
                    -fno-threadsafe-statics -Os -ffunction-sections -fdata-sections \
                    -Wno-unused-parameter

# Cross-compiler for embedded C++
EMBEDDED_CXX = riscv64-unknown-elf-g++

# Base embedded flags (derived from your example command)
EMBEDDED_BASE_FLAGS = -c -MD -MP -Os -march=rv32im -mabi=ilp32 -D__vexriscv__ -g3 \
                      -fomit-frame-pointer -Wall -fno-builtin -fno-stack-protector \
                      -nostdinc

# Embedded include paths (using environment variables)
EMBEDDED_INCLUDES = -I$(POGO_SDK_TOOLS)/include/litex/pythondata-software-picolibc/pythondata_software_picolibc/data/newlib/libc/tinystdio \
                    -I$(POGO_SDK_TOOLS)/include/litex/pythondata-software-picolibc/pythondata_software_picolibc/data/newlib/libc/include \
                    -I$(POGO_SDK_TOOLS)/include/litex/soc/software/libbase \
                    -I$(POGO_SDK_TOOLS)/include/litex/soc/software/include \
                    -I$(POGO_SDK_TOOLS)/include/litex/soc/software \
                    -I$(POGO_SDK_TOOLS)/include/pogobotv3/software/include \
                    -I$(POGO_SDK_TOOLS)/include/pogobotv3/software/include/../libc \
                    -I$(POGO_SDK_TOOLS)/include/litex/soc/cores/cpu/vexriscv \
                    -I. -I$(POGO_SDK_INCS) -I$(POGOSIM_INCLUDE_DIR) \
                    -I/usr/lib/gcc/x86_64-linux-gnu/13/include \
                    -I/usr/lib/gcc/x86_64-linux-gnu/13/include-fixed

# pull in dependency info for *existing* .o files
-include $(OBJECTS_BUILD:.o=.d) $(SIM_OBJECTS:.o=.d) $(DEP_FILES)

$(BUILD_DIR)/bin/%.bin: $(BUILD_DIR)/bin/%.elf
	$(OBJCOPY) -O binary $< $@
	chmod -x $@

$(BUILD_DIR)/bin/firmware.elf: $(OBJECTS_BUILD)
	# Use the cross-compiler (CC) for linking, with C++ standard library if needed
	@if [ -n "$(CPP_OBJECTS_BUILD)" ]; then \
		echo "Linking with cross-compiler, C++ support enabled"; \
		$(CC) $(LDFLAGS) $(EMBEDDED_CXXFLAGS) \
			-T $(POGO_SDK_TOOLS)/linker.ld \
			-N -o $@ \
			$(OBJECTS_BUILD) \
			$(WHOLE_ARCHIVE_FLAG) \
			-Wl,--gc-sections \
			-L$(POGO_SDK_LIBS) -lcompiler_rt -lc -lpogobot; \
	else \
		echo "Linking with C compiler"; \
		$(CC) $(LDFLAGS) \
			-T $(POGO_SDK_TOOLS)/linker.ld \
			-N -o $@ \
			$(OBJECTS_BUILD) \
			$(WHOLE_ARCHIVE_FLAG) \
			-Wl,--gc-sections \
			-L$(POGO_SDK_LIBS) -lcompiler_rt -lc -lpogobot; \
	fi
	chmod -x $@

# Rules for C files
$(BUILD_DIR)/bin/%.o: %.c | $(BUILD_DIR)/bin
	$(compile) -DREAL_ROBOT -DROBOT_CATEGORY=$(ROBOT_CATEGORY)

$(BUILD_DIR)/bin/%.o: $(POGOSIM_SRC_DIR)/%.c | $(BUILD_DIR)/bin
	$(compile) -DREAL_ROBOT -DROBOT_CATEGORY=$(ROBOT_CATEGORY)

# Rules for C++ files - use RISC-V G++ cross-compiler directly
$(BUILD_DIR)/bin/%.o: %.cpp | $(BUILD_DIR)/bin
	$(EMBEDDED_CXX) $(EMBEDDED_BASE_FLAGS) $(EMBEDDED_CXXFLAGS) $(EMBEDDED_INCLUDES) \
		-DREAL_ROBOT -DROBOT_CATEGORY=$(ROBOT_CATEGORY) $< -o $@

$(BUILD_DIR)/bin/%.o: $(POGOSIM_SRC_DIR)/%.cpp | $(BUILD_DIR)/bin
	$(EMBEDDED_CXX) $(EMBEDDED_BASE_FLAGS) $(EMBEDDED_CXXFLAGS) $(EMBEDDED_INCLUDES) \
		-DREAL_ROBOT -DROBOT_CATEGORY=$(ROBOT_CATEGORY) $< -o $@

clean:
	$(RM) -r $(BUILD_DIR) .*~ *~
	$(RM) -f $(SIM_OBJECTS) $(SIM_TARGET)

connect:
	$(POGO_SDK_TOOLS)/litex_term.py --serial-boot --kernel $(BUILD_DIR)/bin/firmware.bin --kernel-adr $(ROM_BASE) --safe $(TTY)

directories: $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all clean connect directories

# Debug target to show what files are being compiled
debug:
	@echo "C sources: $(C_SRCS)"
	@echo "C++ sources: $(CPP_SRCS)"
	@echo "C objects: $(C_OBJECTS_BUILD)"
	@echo "C++ objects: $(CPP_OBJECTS_BUILD)"
	@echo "All objects: $(OBJECTS_BUILD)"

# MODELINE "{{{1
# vim:noexpandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
