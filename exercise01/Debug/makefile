################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm

# All of the sources participating in the build are defined here
-include sources.mk
-include myFirstKernel/subdir.mk
-include gamma/subdir.mk
-include dotProduct/subdir.mk
-include deviceQuery/subdir.mk
-include cudaMallocAndMemcpy/subdir.mk
-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := exercise01
BUILD_ARTIFACT_EXTENSION :=
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: main-build

# Main-build Target
main-build: exercise01

# Tool invocations
exercise01: $(OBJS) $(USER_OBJS) makefile $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/bin/nvcc --cudart=static -ccbin g++ -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o "exercise01" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) exercise01
	-@echo ' '

.PHONY: all clean dependents main-build

-include ../makefile.targets
