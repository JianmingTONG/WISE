cmake_minimum_required(VERSION 3.20)
project(fhecore_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python & pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

if(TARGET OPENFHEpke)
    set(PKE_TARGET OPENFHEpke)
elseif(TARGET OpenFHE::pke)
    set(PKE_TARGET OpenFHE::pke)
else()
    find_package(OpenFHE REQUIRED CONFIG)
    if(TARGET OPENFHEpke)
        set(PKE_TARGET OPENFHEpke)
    elseif(TARGET OpenFHE::pke)
        set(PKE_TARGET OpenFHE::pke)
    else()
        message(FATAL_ERROR "Could not find OpenFHE pke target (OPENFHEpke or OpenFHE::pke).")
    endif()
endif()

set(OFHE_INCLUDE_ROOT "")
get_target_property(_ofhe_incs ${PKE_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
if(_ofhe_incs)
    foreach(p IN LISTS _ofhe_incs)
        if(p MATCHES "/include/?$")
            set(OFHE_INCLUDE_ROOT "${p}")
            break()
        endif()
    endforeach()
    if(NOT OFHE_INCLUDE_ROOT)
        list(GET _ofhe_incs 0 OFHE_INCLUDE_ROOT)
    endif()
endif()
if(NOT OFHE_INCLUDE_ROOT AND DEFINED OpenFHE_DIR)
    get_filename_component(_libdir "${OpenFHE_DIR}" DIRECTORY)
    get_filename_component(_prefix  "${_libdir}" DIRECTORY)
    set(OFHE_INCLUDE_ROOT "${_prefix}/include")
endif()

if(NOT TARGET fheproj)
    add_library(fheproj STATIC
    ../../src/context.cpp
    ../../src/ct_handle.cpp
  )
    set_target_properties(fheproj PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_include_directories(fheproj
    PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/../../include
  )
    if(OFHE_INCLUDE_ROOT)
        target_include_directories(fheproj PRIVATE
      "${OFHE_INCLUDE_ROOT}"
      "${OFHE_INCLUDE_ROOT}/openfhe"
      "${OFHE_INCLUDE_ROOT}/openfhe/pke"
      "${OFHE_INCLUDE_ROOT}/openfhe/core"
      "${OFHE_INCLUDE_ROOT}/openfhe/binfhe"
    )
    endif()

    target_link_libraries(fheproj PUBLIC ${PKE_TARGET})

    if(MSVC)
        target_compile_options(fheproj PRIVATE /O2)
    else()
        target_compile_options(fheproj PRIVATE -O3 -fno-plt)
    endif()

endif()

pybind11_add_module(fhecore MODULE module.cpp)

target_include_directories(fhecore PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../../include
)
if(OFHE_INCLUDE_ROOT)
    target_include_directories(fhecore PRIVATE
    "${OFHE_INCLUDE_ROOT}"
    "${OFHE_INCLUDE_ROOT}/openfhe"
    "${OFHE_INCLUDE_ROOT}/openfhe/pke"
    "${OFHE_INCLUDE_ROOT}/openfhe/core"
    "${OFHE_INCLUDE_ROOT}/openfhe/binfhe"
  )
endif()

target_link_libraries(fhecore PRIVATE
  fheproj
  ${PKE_TARGET}
)

if(MSVC)
    target_compile_options(fhecore PRIVATE /O2)
else()
    target_compile_options(fhecore PRIVATE -O3 -fno-plt)
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(fhecore PROPERTIES BUILD_RPATH "\$ORIGIN")
endif()
get_target_property(_ofhe_lib2 ${PKE_TARGET} IMPORTED_LOCATION_RELEASE)
if(NOT _ofhe_lib2)
    get_target_property(_ofhe_lib2 ${PKE_TARGET} IMPORTED_LOCATION)
endif()
if(_ofhe_lib2)
    get_filename_component(_ofhe_dir2 "${_ofhe_lib2}" DIRECTORY)
    set_property(TARGET fhecore APPEND PROPERTY BUILD_RPATH "${_ofhe_dir2}")
endif()

install(TARGETS fhecore LIBRARY DESTINATION .)

