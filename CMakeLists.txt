cmake_minimum_required(VERSION 3.5)

# Set the project name
project (dgemm)

# Add an executable
file(GLOB SOURCES src/*.cpp)
add_executable(dgemm ${SOURCES})

find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )

find_package(OpenMP)

include_directories(${BLAS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wno-sign-compare -Woverloaded-virtual -Wwrite-strings -Wno-unused")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lcblas")
endif()

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

TARGET_LINK_LIBRARIES(dgemm m blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# By hand: g++ -std=c++11   main.cpp -lcblas
