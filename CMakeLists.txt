cmake_minimum_required(VERSION 3.16)
project(litenp VERSION 0.1.0 LANGUAGES CXX)

include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(LITENP_TOP_LEVEL_DEFAULT ON)
else()
    set(LITENP_TOP_LEVEL_DEFAULT OFF)
endif()

option(LITENP_BUILD_TESTS "Build litenp tests" ${LITENP_TOP_LEVEL_DEFAULT})
option(LITENP_BUILD_BENCHMARKS "Build litenp benchmarks" OFF)
option(LITENP_BUILD_EXAMPLES "Build litenp examples" ${LITENP_TOP_LEVEL_DEFAULT})
option(LITENP_USE_OPENMP "Use OpenMP for parallel CPU kernels when available" OFF)
option(LITENP_USE_CBLAS "Use CBLAS/OpenBLAS for matmul when available" OFF)
option(LITENP_NATIVE_ARCH "Compile litenp consumers with -march=native on GNU/Clang" OFF)

add_library(litenp INTERFACE)
add_library(litenp::litenp ALIAS litenp)

target_compile_features(litenp INTERFACE cxx_std_17)
target_include_directories(
    litenp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(LITENP_NATIVE_ARCH)
    target_compile_options(
        litenp
        INTERFACE
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-march=native>
    )
endif()

if(LITENP_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(litenp INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(litenp INTERFACE LITENP_USE_OPENMP=1)
    endif()
endif()

if(LITENP_USE_CBLAS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS QUIET openblas)
    endif()

    if(OPENBLAS_FOUND)
        target_include_directories(litenp INTERFACE ${OPENBLAS_INCLUDE_DIRS})
        target_link_directories(litenp INTERFACE ${OPENBLAS_LIBRARY_DIRS})
        target_link_libraries(litenp INTERFACE ${OPENBLAS_LIBRARIES})
        target_compile_definitions(litenp INTERFACE LITENP_HAS_CBLAS=1)
    else()
        find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas-pthread x86_64-linux-gnu)
        find_library(CBLAS_LIBRARY NAMES openblas cblas blas)
        if(CBLAS_INCLUDE_DIR AND CBLAS_LIBRARY)
            target_include_directories(litenp INTERFACE "${CBLAS_INCLUDE_DIR}")
            target_link_libraries(litenp INTERFACE "${CBLAS_LIBRARY}")
            target_compile_definitions(litenp INTERFACE LITENP_HAS_CBLAS=1)
        endif()
    endif()
endif()

if(LITENP_BUILD_TESTS)
    enable_testing()
    add_executable(litenp_tests tests/test_litenp.cpp)
    target_link_libraries(litenp_tests PRIVATE litenp::litenp)
    add_test(NAME litenp_tests COMMAND litenp_tests)
endif()

if(LITENP_BUILD_BENCHMARKS)
    add_executable(litenp_bench benchmarks/bench_litenp.cpp)
    target_link_libraries(litenp_bench PRIVATE litenp::litenp)

    find_path(
        EIGEN3_INCLUDE_DIR
        Eigen/Core
        HINTS
            "${CMAKE_CURRENT_SOURCE_DIR}/../data/eigen-3.4.0"
            "/usr/include/eigen3"
            "/usr/local/include/eigen3"
    )
    if(EIGEN3_INCLUDE_DIR)
        target_include_directories(litenp_bench PRIVATE "${EIGEN3_INCLUDE_DIR}")
        target_compile_definitions(litenp_bench PRIVATE LITENP_HAS_EIGEN=1)
    endif()

    find_package(Torch QUIET)
    if(Torch_FOUND)
        target_link_libraries(litenp_bench PRIVATE "${TORCH_LIBRARIES}")
        target_compile_definitions(litenp_bench PRIVATE LITENP_HAS_TORCH=1)
    endif()
endif()

if(LITENP_BUILD_EXAMPLES)
    add_executable(litenp_basic examples/basic.cpp)
    target_link_libraries(litenp_basic PRIVATE litenp::litenp)
endif()

configure_package_config_file(
    cmake/litenpConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/litenpConfig.cmake"
    INSTALL_DESTINATION lib/cmake/litenp
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/litenpConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS litenp EXPORT litenpTargets)
install(EXPORT litenpTargets NAMESPACE litenp:: DESTINATION lib/cmake/litenp)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/litenpConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/litenpConfigVersion.cmake"
    DESTINATION lib/cmake/litenp
)
