# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dwz/my-work/cupc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dwz/my-work/cupc/build

# Include any dependencies generated for this target.
include t0_hello/CMakeFiles/t0_hello.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include t0_hello/CMakeFiles/t0_hello.dir/compiler_depend.make

# Include the progress variables for this target.
include t0_hello/CMakeFiles/t0_hello.dir/progress.make

# Include the compile flags for this target's objects.
include t0_hello/CMakeFiles/t0_hello.dir/flags.make

t0_hello/CMakeFiles/t0_hello.dir/codegen:
.PHONY : t0_hello/CMakeFiles/t0_hello.dir/codegen

t0_hello/CMakeFiles/t0_hello.dir/hello.cu.o: t0_hello/CMakeFiles/t0_hello.dir/flags.make
t0_hello/CMakeFiles/t0_hello.dir/hello.cu.o: /home/dwz/my-work/cupc/t0_hello/hello.cu
t0_hello/CMakeFiles/t0_hello.dir/hello.cu.o: t0_hello/CMakeFiles/t0_hello.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/dwz/my-work/cupc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object t0_hello/CMakeFiles/t0_hello.dir/hello.cu.o"
	cd /home/dwz/my-work/cupc/build/t0_hello && /opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT t0_hello/CMakeFiles/t0_hello.dir/hello.cu.o -MF CMakeFiles/t0_hello.dir/hello.cu.o.d -x cu -c /home/dwz/my-work/cupc/t0_hello/hello.cu -o CMakeFiles/t0_hello.dir/hello.cu.o

t0_hello/CMakeFiles/t0_hello.dir/hello.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/t0_hello.dir/hello.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

t0_hello/CMakeFiles/t0_hello.dir/hello.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/t0_hello.dir/hello.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target t0_hello
t0_hello_OBJECTS = \
"CMakeFiles/t0_hello.dir/hello.cu.o"

# External object files for target t0_hello
t0_hello_EXTERNAL_OBJECTS =

t0_hello/t0_hello: t0_hello/CMakeFiles/t0_hello.dir/hello.cu.o
t0_hello/t0_hello: t0_hello/CMakeFiles/t0_hello.dir/build.make
t0_hello/t0_hello: t0_hello/CMakeFiles/t0_hello.dir/compiler_depend.ts
t0_hello/t0_hello: t0_hello/CMakeFiles/t0_hello.dir/linkLibs.rsp
t0_hello/t0_hello: t0_hello/CMakeFiles/t0_hello.dir/objects1.rsp
t0_hello/t0_hello: t0_hello/CMakeFiles/t0_hello.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/dwz/my-work/cupc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable t0_hello"
	cd /home/dwz/my-work/cupc/build/t0_hello && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/t0_hello.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
t0_hello/CMakeFiles/t0_hello.dir/build: t0_hello/t0_hello
.PHONY : t0_hello/CMakeFiles/t0_hello.dir/build

t0_hello/CMakeFiles/t0_hello.dir/clean:
	cd /home/dwz/my-work/cupc/build/t0_hello && $(CMAKE_COMMAND) -P CMakeFiles/t0_hello.dir/cmake_clean.cmake
.PHONY : t0_hello/CMakeFiles/t0_hello.dir/clean

t0_hello/CMakeFiles/t0_hello.dir/depend:
	cd /home/dwz/my-work/cupc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dwz/my-work/cupc /home/dwz/my-work/cupc/t0_hello /home/dwz/my-work/cupc/build /home/dwz/my-work/cupc/build/t0_hello /home/dwz/my-work/cupc/build/t0_hello/CMakeFiles/t0_hello.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : t0_hello/CMakeFiles/t0_hello.dir/depend

