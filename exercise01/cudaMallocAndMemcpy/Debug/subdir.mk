################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cudaMallocAndMemcpy.cu 

OBJS += \
./cudaMallocAndMemcpy.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/bin/nvcc --device-debug --debug -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -ccbin g++ -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

cudaMallocAndMemcpy.o: /usr/include/stdc-predef.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/cuda_runtime.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/host_config.h /usr/include/features.h /usr/include/features-time64.h /usr/include/x86_64-linux-gnu/bits/wordsize.h /usr/include/x86_64-linux-gnu/bits/timesize.h /usr/include/x86_64-linux-gnu/sys/cdefs.h /usr/include/x86_64-linux-gnu/bits/long-double.h /usr/include/x86_64-linux-gnu/gnu/stubs.h /usr/include/x86_64-linux-gnu/gnu/stubs-64.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/builtin_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/device_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/host_defines.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/driver_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/vector_types.h /usr/lib/gcc/x86_64-linux-gnu/11/include/limits.h /usr/lib/gcc/x86_64-linux-gnu/11/include/syslimits.h /usr/include/limits.h /usr/include/x86_64-linux-gnu/bits/libc-header-start.h /usr/include/x86_64-linux-gnu/bits/posix1_lim.h /usr/include/x86_64-linux-gnu/bits/local_lim.h /usr/include/linux/limits.h /usr/include/x86_64-linux-gnu/bits/pthread_stack_min-dynamic.h /usr/include/x86_64-linux-gnu/bits/posix2_lim.h /usr/include/x86_64-linux-gnu/bits/xopen_lim.h /usr/include/x86_64-linux-gnu/bits/uio_lim.h /usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/surface_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/texture_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/library_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/channel_descriptor.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/cuda_runtime_api.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/cuda_device_runtime_api.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/driver_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/vector_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/vector_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/common_functions.h /usr/include/string.h /usr/include/x86_64-linux-gnu/bits/types/locale_t.h /usr/include/x86_64-linux-gnu/bits/types/__locale_t.h /usr/include/strings.h /usr/include/time.h /usr/include/x86_64-linux-gnu/bits/time.h /usr/include/x86_64-linux-gnu/bits/types.h /usr/include/x86_64-linux-gnu/bits/typesizes.h /usr/include/x86_64-linux-gnu/bits/time64.h /usr/include/x86_64-linux-gnu/bits/timex.h /usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h /usr/include/x86_64-linux-gnu/bits/types/clock_t.h /usr/include/x86_64-linux-gnu/bits/types/time_t.h /usr/include/x86_64-linux-gnu/bits/types/struct_tm.h /usr/include/x86_64-linux-gnu/bits/types/struct_timespec.h /usr/include/x86_64-linux-gnu/bits/endian.h /usr/include/x86_64-linux-gnu/bits/endianness.h /usr/include/x86_64-linux-gnu/bits/types/clockid_t.h /usr/include/x86_64-linux-gnu/bits/types/timer_t.h /usr/include/x86_64-linux-gnu/bits/types/struct_itimerspec.h /usr/include/c++/11/new /usr/include/x86_64-linux-gnu/c++/11/bits/c++config.h /usr/include/x86_64-linux-gnu/c++/11/bits/os_defines.h /usr/include/x86_64-linux-gnu/c++/11/bits/cpu_defines.h /usr/include/c++/11/pstl/pstl_config.h /usr/include/c++/11/bits/exception.h /usr/include/stdio.h /usr/lib/gcc/x86_64-linux-gnu/11/include/stdarg.h /usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h /usr/include/x86_64-linux-gnu/bits/types/__mbstate_t.h /usr/include/x86_64-linux-gnu/bits/types/__fpos64_t.h /usr/include/x86_64-linux-gnu/bits/types/__FILE.h /usr/include/x86_64-linux-gnu/bits/types/FILE.h /usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h /usr/include/x86_64-linux-gnu/bits/types/cookie_io_functions_t.h /usr/include/x86_64-linux-gnu/bits/stdio_lim.h /usr/include/x86_64-linux-gnu/bits/floatn.h /usr/include/x86_64-linux-gnu/bits/floatn-common.h /usr/include/c++/11/stdlib.h /usr/include/c++/11/cstdlib /usr/include/stdlib.h /usr/include/x86_64-linux-gnu/bits/waitflags.h /usr/include/x86_64-linux-gnu/bits/waitstatus.h /usr/include/x86_64-linux-gnu/sys/types.h /usr/include/x86_64-linux-gnu/bits/stdint-intn.h /usr/include/endian.h /usr/include/x86_64-linux-gnu/bits/byteswap.h /usr/include/x86_64-linux-gnu/bits/uintn-identity.h /usr/include/x86_64-linux-gnu/sys/select.h /usr/include/x86_64-linux-gnu/bits/select.h /usr/include/x86_64-linux-gnu/bits/types/sigset_t.h /usr/include/x86_64-linux-gnu/bits/types/__sigset_t.h /usr/include/x86_64-linux-gnu/bits/pthreadtypes.h /usr/include/x86_64-linux-gnu/bits/thread-shared-types.h /usr/include/x86_64-linux-gnu/bits/pthreadtypes-arch.h /usr/include/x86_64-linux-gnu/bits/atomic_wide_counter.h /usr/include/x86_64-linux-gnu/bits/struct_mutex.h /usr/include/x86_64-linux-gnu/bits/struct_rwlock.h /usr/include/alloca.h /usr/include/x86_64-linux-gnu/bits/stdlib-float.h /usr/include/c++/11/bits/std_abs.h /usr/include/assert.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/math_functions.h /usr/include/c++/11/math.h /usr/include/c++/11/cmath /usr/include/c++/11/bits/cpp_type_traits.h /usr/include/c++/11/ext/type_traits.h /usr/include/math.h /usr/include/x86_64-linux-gnu/bits/math-vector.h /usr/include/x86_64-linux-gnu/bits/libm-simd-decl-stubs.h /usr/include/x86_64-linux-gnu/bits/flt-eval-method.h /usr/include/x86_64-linux-gnu/bits/fp-logb.h /usr/include/x86_64-linux-gnu/bits/fp-fast.h /usr/include/x86_64-linux-gnu/bits/mathcalls-helper-functions.h /usr/include/x86_64-linux-gnu/bits/mathcalls.h /usr/include/x86_64-linux-gnu/bits/mathcalls-narrow.h /usr/include/x86_64-linux-gnu/bits/iscanonical.h /usr/include/c++/11/bits/specfun.h /usr/include/c++/11/bits/stl_algobase.h /usr/include/c++/11/bits/functexcept.h /usr/include/c++/11/bits/exception_defines.h /usr/include/c++/11/ext/numeric_traits.h /usr/include/c++/11/bits/stl_pair.h /usr/include/c++/11/bits/move.h /usr/include/c++/11/type_traits /usr/include/c++/11/bits/stl_iterator_base_types.h /usr/include/c++/11/bits/stl_iterator_base_funcs.h /usr/include/c++/11/bits/concept_check.h /usr/include/c++/11/debug/assertions.h /usr/include/c++/11/bits/stl_iterator.h /usr/include/c++/11/bits/ptr_traits.h /usr/include/c++/11/debug/debug.h /usr/include/c++/11/bits/predefined_ops.h /usr/include/c++/11/limits /usr/include/c++/11/tr1/gamma.tcc /usr/include/c++/11/tr1/special_function_util.h /usr/include/c++/11/tr1/bessel_function.tcc /usr/include/c++/11/tr1/beta_function.tcc /usr/include/c++/11/tr1/ell_integral.tcc /usr/include/c++/11/tr1/exp_integral.tcc /usr/include/c++/11/tr1/hypergeometric.tcc /usr/include/c++/11/tr1/legendre_function.tcc /usr/include/c++/11/tr1/modified_bessel_func.tcc /usr/include/c++/11/tr1/poly_hermite.tcc /usr/include/c++/11/tr1/poly_laguerre.tcc /usr/include/c++/11/tr1/riemann_zeta.tcc /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/math_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/cuda_surface_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/cuda_texture_types.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/device_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/device_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/device_atomic_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/device_atomic_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/device_double_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/device_double_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_20_atomic_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_20_atomic_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_32_atomic_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_32_atomic_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_35_atomic_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_60_atomic_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_60_atomic_functions.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_20_intrinsics.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_20_intrinsics.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_30_intrinsics.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_30_intrinsics.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_32_intrinsics.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_32_intrinsics.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_35_intrinsics.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_61_intrinsics.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/sm_61_intrinsics.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/sm_70_rt.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/sm_70_rt.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/sm_80_rt.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/sm_80_rt.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/sm_90_rt.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/sm_90_rt.hpp /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/surface_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/texture_fetch_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/texture_indirect_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/surface_indirect_functions.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/crt/cudacc_ext.h /graphics/opt/opt_Ubuntu22.04/cuda/toolkit_11.8.0/cuda/targets/x86_64-linux/include/device_launch_parameters.h /usr/include/c++/11/utility /usr/include/c++/11/bits/stl_relops.h /usr/include/c++/11/initializer_list /usr/include/c++/11/cassert /usr/include/c++/11/cstdio


clean: clean--2e-

clean--2e-:
	-$(RM) ./cudaMallocAndMemcpy.o

.PHONY: clean--2e-

