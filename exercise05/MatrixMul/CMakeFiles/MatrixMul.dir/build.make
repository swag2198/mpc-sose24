# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /graphics/opt/opt_Ubuntu22.04/python/3.10/opt-packages/local/lib/python3.10/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /graphics/opt/opt_Ubuntu22.04/python/3.10/opt-packages/local/lib/python3.10/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/haldarsw/ex34/exercise05/MatrixMul

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/haldarsw/ex34/exercise05/MatrixMul

# Include any dependencies generated for this target.
include CMakeFiles/MatrixMul.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/MatrixMul.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/MatrixMul.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MatrixMul.dir/flags.make

CMakeFiles/MatrixMul.dir/main.cu.o: CMakeFiles/MatrixMul.dir/flags.make
CMakeFiles/MatrixMul.dir/main.cu.o: CMakeFiles/MatrixMul.dir/includes_CUDA.rsp
CMakeFiles/MatrixMul.dir/main.cu.o: main.cu
CMakeFiles/MatrixMul.dir/main.cu.o: CMakeFiles/MatrixMul.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haldarsw/ex34/exercise05/MatrixMul/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/MatrixMul.dir/main.cu.o"
	/graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/MatrixMul.dir/main.cu.o -MF CMakeFiles/MatrixMul.dir/main.cu.o.d -x cu -c /home/haldarsw/ex34/exercise05/MatrixMul/main.cu -o CMakeFiles/MatrixMul.dir/main.cu.o

CMakeFiles/MatrixMul.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/MatrixMul.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/MatrixMul.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/MatrixMul.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target MatrixMul
MatrixMul_OBJECTS = \
"CMakeFiles/MatrixMul.dir/main.cu.o"

# External object files for target MatrixMul
MatrixMul_EXTERNAL_OBJECTS =

MatrixMul: CMakeFiles/MatrixMul.dir/main.cu.o
MatrixMul: CMakeFiles/MatrixMul.dir/build.make
MatrixMul: CMakeFiles/MatrixMul.dir/linkLibs.rsp
MatrixMul: CMakeFiles/MatrixMul.dir/objects1.rsp
MatrixMul: CMakeFiles/MatrixMul.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/haldarsw/ex34/exercise05/MatrixMul/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable MatrixMul"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MatrixMul.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MatrixMul.dir/build: MatrixMul
.PHONY : CMakeFiles/MatrixMul.dir/build

CMakeFiles/MatrixMul.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MatrixMul.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MatrixMul.dir/clean

CMakeFiles/MatrixMul.dir/depend:
	cd /home/haldarsw/ex34/exercise05/MatrixMul && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/haldarsw/ex34/exercise05/MatrixMul /home/haldarsw/ex34/exercise05/MatrixMul /home/haldarsw/ex34/exercise05/MatrixMul /home/haldarsw/ex34/exercise05/MatrixMul /home/haldarsw/ex34/exercise05/MatrixMul/CMakeFiles/MatrixMul.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MatrixMul.dir/depend

