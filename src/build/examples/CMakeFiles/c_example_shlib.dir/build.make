# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/szc/pcm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szc/pcm/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/c_example_shlib.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/c_example_shlib.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/c_example_shlib.dir/flags.make

examples/CMakeFiles/c_example_shlib.dir/c_example.c.o: examples/CMakeFiles/c_example_shlib.dir/flags.make
examples/CMakeFiles/c_example_shlib.dir/c_example.c.o: ../examples/c_example.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/szc/pcm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/CMakeFiles/c_example_shlib.dir/c_example.c.o"
	cd /home/szc/pcm/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/c_example_shlib.dir/c_example.c.o   -c /home/szc/pcm/examples/c_example.c

examples/CMakeFiles/c_example_shlib.dir/c_example.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/c_example_shlib.dir/c_example.c.i"
	cd /home/szc/pcm/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/szc/pcm/examples/c_example.c > CMakeFiles/c_example_shlib.dir/c_example.c.i

examples/CMakeFiles/c_example_shlib.dir/c_example.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/c_example_shlib.dir/c_example.c.s"
	cd /home/szc/pcm/build/examples && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/szc/pcm/examples/c_example.c -o CMakeFiles/c_example_shlib.dir/c_example.c.s

examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.requires:

.PHONY : examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.requires

examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.provides: examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.requires
	$(MAKE) -f examples/CMakeFiles/c_example_shlib.dir/build.make examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.provides.build
.PHONY : examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.provides

examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.provides.build: examples/CMakeFiles/c_example_shlib.dir/c_example.c.o


# Object files for target c_example_shlib
c_example_shlib_OBJECTS = \
"CMakeFiles/c_example_shlib.dir/c_example.c.o"

# External object files for target c_example_shlib
c_example_shlib_EXTERNAL_OBJECTS =

bin/examples/c_example_shlib: examples/CMakeFiles/c_example_shlib.dir/c_example.c.o
bin/examples/c_example_shlib: examples/CMakeFiles/c_example_shlib.dir/build.make
bin/examples/c_example_shlib: lib/libpcm.so
bin/examples/c_example_shlib: examples/CMakeFiles/c_example_shlib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/szc/pcm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../bin/examples/c_example_shlib"
	cd /home/szc/pcm/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c_example_shlib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/c_example_shlib.dir/build: bin/examples/c_example_shlib

.PHONY : examples/CMakeFiles/c_example_shlib.dir/build

examples/CMakeFiles/c_example_shlib.dir/requires: examples/CMakeFiles/c_example_shlib.dir/c_example.c.o.requires

.PHONY : examples/CMakeFiles/c_example_shlib.dir/requires

examples/CMakeFiles/c_example_shlib.dir/clean:
	cd /home/szc/pcm/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/c_example_shlib.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/c_example_shlib.dir/clean

examples/CMakeFiles/c_example_shlib.dir/depend:
	cd /home/szc/pcm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szc/pcm /home/szc/pcm/examples /home/szc/pcm/build /home/szc/pcm/build/examples /home/szc/pcm/build/examples/CMakeFiles/c_example_shlib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/c_example_shlib.dir/depend

