cmake_minimum_required(VERSION 3.10)
project(AREC3D VERSION 1.0)


set(PROJECT_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC_PATH "${PROJECT_PATH}/AREC3D")
set(INCLUDE_PATH "${PROJECT_PATH}/AREC3D")
set(MAIN_NAME "${PROJECT_NAME}_main")
set(LIB_PATH "${PROJECT_BINARY_DIR}/lib")
set(LIB_NAME "${PROJECT_NAME}")


if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-long-long -pedantic")
endif()

set(MAIN_SRC
  "${SRC_PATH}/AREC3D.cpp"
)
set(SRC
    "${SRC_PATH}/ReadAndDistAngles.cpp"
    "${SRC_PATH}/align2dstack.cpp"
    "${SRC_PATH}/arec3dutil.cpp"
    "${SRC_PATH}/arecImage.cpp"
    "${SRC_PATH}/arecImageIO.cpp"
    "${SRC_PATH}/arecImageIO_mpi.cpp"
    "${SRC_PATH}/areccgls.cpp"
    "${SRC_PATH}/areccgls_KB.cpp"
    "${SRC_PATH}/areccgls_stat.cpp"
    "${SRC_PATH}/arecproject.cpp"
    "${SRC_PATH}/arecsirt.cpp"
    "${SRC_PATH}/arecsirt_KB.cpp"
    "${SRC_PATH}/imagetools.cpp"
    
)

include_directories(AFTER "${INCLUDE_PATH}")
add_library(${LIB_NAME} ${SRC})
add_executable(${MAIN_NAME} ${MAIN_SRC})


find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_C_FLAGS "${CMAKE_FLAGS} ${MPI_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")

target_link_libraries(${LIB_NAME} PRIVATE FFTW3::fftw3f ${MPI_LIBRARIES})
target_link_libraries(${MAIN_NAME} PRIVATE ${LIB_NAME} FFTW3::fftw3 ${MPI_LIBRARIES} )

