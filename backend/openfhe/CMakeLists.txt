cmake_minimum_required(VERSION 3.20)
project(fheproj LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenFHE CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)

add_subdirectory(src)

target_include_directories(fheproj PUBLIC
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/binfhe
)
