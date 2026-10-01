cmake_minimum_required(VERSION 3.24)
project(embedded_jev_prism_runtime_v1 LANGUAGES CXX)

set(PRISM_SOURCE_DIR "" CACHE PATH "Existing pinned Prism source checkout")
set(PRISM_GGML_LIBRARY_DIR "" CACHE PATH "Existing pinned GGML CPU/base library directory")
set(JEV_PRISM_REVISION "842b1880415d6f508f03b789e5ce70194def7bfd")
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

if(NOT IS_DIRECTORY "${PRISM_SOURCE_DIR}/src" OR NOT IS_DIRECTORY "${PRISM_GGML_LIBRARY_DIR}")
    message(FATAL_ERROR "Set PRISM_SOURCE_DIR and PRISM_GGML_LIBRARY_DIR to the existing cache")
endif()
file(REAL_PATH "${PRISM_SOURCE_DIR}" PRISM_SOURCE_DIR)
file(REAL_PATH "${PRISM_GGML_LIBRARY_DIR}" PRISM_GGML_LIBRARY_DIR)
get_filename_component(PRISM_EXISTING_BUILD_DIR "${PRISM_GGML_LIBRARY_DIR}" DIRECTORY)
cmake_path(IS_PREFIX PRISM_SOURCE_DIR "${CMAKE_BINARY_DIR}" NORMALIZE JEV_INSIDE_SOURCE)
cmake_path(IS_PREFIX PRISM_EXISTING_BUILD_DIR "${CMAKE_BINARY_DIR}" NORMALIZE JEV_INSIDE_EXISTING_BUILD)
if(JEV_INSIDE_SOURCE OR JEV_INSIDE_EXISTING_BUILD)
    message(FATAL_ERROR "Use a separate build directory; the pinned source/build must remain unchanged")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PRISM_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE JEV_ACTUAL_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JEV_REVISION_STATUS)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PRISM_SOURCE_DIR}" status --porcelain --untracked-files=no
    OUTPUT_VARIABLE JEV_SOURCE_CHANGES OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE JEV_STATUS_STATUS)
if(NOT JEV_REVISION_STATUS EQUAL 0 OR NOT JEV_ACTUAL_REVISION STREQUAL JEV_PRISM_REVISION OR
   NOT JEV_STATUS_STATUS EQUAL 0 OR NOT JEV_SOURCE_CHANGES STREQUAL "")
    message(FATAL_ERROR "Prism checkout must be clean at ${JEV_PRISM_REVISION}")
endif()
if(CMAKE_CROSSCOMPILING OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "This versioned runtime control supports native x86-64 AVX2 only")
endif()
include(CheckCXXSourceRuns)
check_cxx_source_runs("int main() { return __builtin_cpu_supports(\"avx2\") ? 0 : 1; }" JEV_HOST_AVX2)
if(NOT JEV_HOST_AVX2)
    message(FATAL_ERROR "The runtime control requires an AVX2-capable host")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/bin;${PRISM_GGML_LIBRARY_DIR}")

foreach(JEV_DEPENDENCY IN ITEMS ggml-base ggml-cpu)
    if(NOT EXISTS "${PRISM_GGML_LIBRARY_DIR}/lib${JEV_DEPENDENCY}.so")
        message(FATAL_ERROR "Missing cached lib${JEV_DEPENDENCY}.so")
    endif()
    add_library(${JEV_DEPENDENCY} SHARED IMPORTED GLOBAL)
    set_target_properties(${JEV_DEPENDENCY} PROPERTIES
        IMPORTED_LOCATION "${PRISM_GGML_LIBRARY_DIR}/lib${JEV_DEPENDENCY}.so"
        INTERFACE_INCLUDE_DIRECTORIES "${PRISM_SOURCE_DIR}/ggml/include")
endforeach()
add_library(ggml SHARED "${PRISM_SOURCE_DIR}/ggml/src/ggml-backend-reg.cpp"
    "${PRISM_SOURCE_DIR}/ggml/src/ggml-backend-dl.cpp")
target_compile_definitions(ggml PRIVATE GGML_USE_CPU GGML_BUILD PUBLIC GGML_SHARED)
target_include_directories(ggml PRIVATE "${PRISM_SOURCE_DIR}/ggml/src")
target_link_libraries(ggml PUBLIC ggml-base ggml-cpu Threads::Threads ${CMAKE_DL_LIBS})

set(BUILD_SHARED_LIBS ON)
set(LLAMA_VERSION_MAJOR 0)
set(LLAMA_VERSION_BASE "0.2.0")
set(LLAMA_VERSION "${LLAMA_VERSION_BASE}-jev-prism-runtime-v1-${JEV_PRISM_REVISION}")
set(LLAMA_BUILD_COMMIT "${JEV_PRISM_REVISION}")
file(READ "${PRISM_SOURCE_DIR}/cmake/common.cmake" JEV_LLAMA_FLAGS)
string(REPLACE "include(\"ggml/cmake/common.cmake\")"
    "include(\"${PRISM_SOURCE_DIR}/ggml/cmake/common.cmake\")" JEV_LLAMA_FLAGS "${JEV_LLAMA_FLAGS}")
cmake_language(EVAL CODE "${JEV_LLAMA_FLAGS}")
add_subdirectory("${PRISM_SOURCE_DIR}/src" "${CMAKE_BINARY_DIR}/llama")

add_library(prism_group_scale SHARED prism_group_scale.cpp bitnet_group_scale.cpp)
target_compile_options(prism_group_scale PRIVATE -mavx2)
target_include_directories(prism_group_scale PRIVATE "${PRISM_SOURCE_DIR}/include"
    "${PRISM_SOURCE_DIR}/ggml/src" "${PRISM_SOURCE_DIR}/ggml/src/ggml-cpu")
target_link_libraries(prism_group_scale PRIVATE ggml-cpu ggml-base ${CMAKE_DL_LIBS})
add_executable(prism_bitnet_loader_control prism_bitnet_loader_control.cpp)
target_compile_definitions(prism_bitnet_loader_control PRIVATE JEV_TEST_REAL_LOADER=1 JEV_TEST_FULL_RUNTIME=1)
target_include_directories(prism_bitnet_loader_control PRIVATE "${PRISM_SOURCE_DIR}/src")
target_link_libraries(prism_bitnet_loader_control PRIVATE prism_group_scale llama Threads::Threads)

enable_testing()
add_test(NAME jev_runtime_early_init COMMAND prism_bitnet_loader_control)
add_test(NAME jev_runtime_late_init COMMAND prism_bitnet_loader_control --late)