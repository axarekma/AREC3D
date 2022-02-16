cmake_minimum_required(VERSION 3.10)
project(Prealign VERSION 1.0)


if (NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT environment variable not defined")
endif()

set(VCPKG_ROOT $ENV{VCPKG_ROOT})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(FFTW_ROOT "${VCPKG_ROOT}/packages/fftw3_x64-windows/share")
else()
    set(FFTW_ROOT "${VCPKG_ROOT}/packages/fftw3_x64-linux/share")
endif()

message("set fftwroot ${FFTW_ROOT}")

set(FFTW3_DIR  ${FFTW_ROOT}/fftw3)
set(FFTW3f_DIR ${FFTW_ROOT}/fftw3f)
set(FFTW3l_DIR ${FFTW_ROOT}/fftw3l)


set(PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC_PATH "${PROJECT_PATH}/Prealign_2017")
set(INCLUDE_PATH "${PROJECT_PATH}/Prealign_2017")
set(MAIN_NAME "${PROJECT_NAME}_main")
set(LIB_PATH "${PROJECT_BINARY_DIR}/lib")
set(LIB_NAME "${PROJECT_NAME}")


set(MAIN_SRC
  "${SRC_PATH}/Prealign_2017.cpp"
)
set(SRC
  "${SRC_PATH}/mpfit.cpp"
  "${SRC_PATH}/TubeAlign.cpp"
  "${SRC_PATH}/TubeAlign_utils.cpp"
)

include_directories(AFTER "${INCLUDE_PATH}")
add_library(${LIB_NAME} ${SRC})
add_executable(${MAIN_NAME} ${MAIN_SRC})


find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)
target_link_libraries(${LIB_NAME} PRIVATE FFTW3::fftw3f)
target_link_libraries(${MAIN_NAME} PRIVATE ${LIB_NAME} FFTW3::fftw3)

