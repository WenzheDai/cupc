cmake_minimum_required(VERSION 3.28)

project(cupc LANGUAGES CUDA CXX)

if(CUDA_ENABLE)
    enable_language(CUDA)
endif()

include(CheckLanguage)
check_language(CUDA)

include_directories(include)

add_subdirectory(t0_hello)
add_subdirectory(t1_check_dimension)
add_subdirectory(t2_grid_block)
add_subdirectory(t3_sum_array)
add_subdirectory(t_tid)
add_subdirectory(t4_sum_mat)
add_subdirectory(t5_reduceInteger)
add_subdirectory(t6_checkGlobalVariable)
add_subdirectory(t7_pine_memory)
add_subdirectory(t8_zero_copy_memory)
add_subdirectory(t9_UVA)
add_subdirectory(t11_transformmat)
add_subdirectory(t12_uniform_memory_managed)


