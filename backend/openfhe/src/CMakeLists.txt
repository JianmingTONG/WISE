add_library(fheproj STATIC
    context.cpp
    ct_handle.cpp
)

set(PROJ_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(PROJ_INCLUDE_DIR_FALLBACK "${CMAKE_CURRENT_LIST_DIR}/../include")

target_include_directories(fheproj
  PUBLIC
    "${PROJ_INCLUDE_DIR}"
    "${PROJ_INCLUDE_DIR_FALLBACK}"
    ${CMAKE_SOURCE_DIR}/include
)

if(TARGET OpenFHE::openfhe)
    target_link_libraries(fheproj PUBLIC OpenFHE::openfhe)
elseif(TARGET OPENFHEpke)
    target_link_libraries(fheproj PUBLIC OPENFHEpke)
    get_target_property(_ofhe_inc OPENFHEpke INTERFACE_INCLUDE_DIRECTORIES)
    if(_ofhe_inc) target_include_directories(fheproj PUBLIC ${_ofhe_inc})
    else()
        get_filename_component(_libdir "${OpenFHE_DIR}" DIRECTORY)
        get_filename_component(_prefix  "${_libdir}" DIRECTORY)
        target_include_directories(fheproj PUBLIC "${_prefix}/include")
    endif()
else()
    message(FATAL_ERROR "OpenFHE target not found. Check -DOpenFHE_DIR=...")
endif()

if(MSVC)
    target_compile_options(fheproj PRIVATE /O2)
else()
    target_compile_options(fheproj PRIVATE -O3 -fno-plt)
endif()

get_target_property(_ofhe_lib OPENFHEpke IMPORTED_LOCATION_RELEASE)
if(NOT _ofhe_lib)
    get_target_property(_ofhe_lib OPENFHEpke IMPORTED_LOCATION)
endif()
if(_ofhe_lib)
    get_filename_component(_ofhe_dir "${_ofhe_lib}" DIRECTORY)
    set_property(TARGET fheproj PROPERTY BUILD_RPATH "${_ofhe_dir}")
endif()

