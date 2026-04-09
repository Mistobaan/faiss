# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyfaiss
  DESCRIPTION "Python bindings for faiss."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

# Find Python early so that Python_SOABI is available for PEP 3149 naming.
# Development.Module is sufficient for extension modules (provides Python::Module,
# Python_INCLUDE_DIRS, and Python_SOABI) without requiring the full embedding
# library (Development.Embed / Python_LIBRARIES).
find_package(Python REQUIRED
  COMPONENTS Interpreter Development.Module NumPy
)

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swigfaiss source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
  if(FAISS_ENABLE_SVS)
    set_property(SOURCE ${source} APPEND PROPERTY
      SWIG_FLAGS -DFAISS_ENABLE_SVS -DFAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION}
    )
  endif()
  if(FAISS_OPT_LEVEL STREQUAL "dd")
    set_property(SOURCE ${source} APPEND PROPERTY
      SWIG_FLAGS -DFAISS_ENABLE_DD
    )
  endif()
  if(FAISS_ENABLE_GPU)
    set_source_files_properties(${source} PROPERTIES
      COMPILE_DEFINITIONS GPU_WRAPPER
    )
    if (FAISS_ENABLE_ROCM)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_ROCM
      )
    endif()
    if (FAISS_ENABLE_CUVS)
      set_property(SOURCE ${source} APPEND PROPERTY
        COMPILE_DEFINITIONS FAISS_ENABLE_CUVS
      )
    endif()
  endif()
  if(FAISS_ENABLE_METAL)
    set_property(SOURCE ${source} APPEND PROPERTY
      COMPILE_DEFINITIONS METAL_WRAPPER
    )
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx2.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_avx512_spr.swig COPYONLY)
configure_file(swigfaiss.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigfaiss_sve.swig COPYONLY)

configure_swigfaiss(swigfaiss.swig)
configure_swigfaiss(swigfaiss_avx2.swig)
configure_swigfaiss(swigfaiss_avx512.swig)
configure_swigfaiss(swigfaiss_avx512_spr.swig)
configure_swigfaiss(swigfaiss_sve.swig)
configure_swigfaiss(faiss_example_external_module.swig)

if(TARGET faiss)
  # Manually add headers as extra dependencies of swigfaiss.
  set(SWIG_MODULE_swigfaiss_EXTRA_DEPS)
  foreach(h ${FAISS_HEADERS})
    list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
    list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/${h}")
  endforeach()
  foreach(h ${FAISS_GPU_HEADERS})
    list(APPEND SWIG_MODULE_swigfaiss_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx2_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_avx512_spr_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_swigfaiss_sve_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
    list(APPEND SWIG_MODULE_faiss_example_external_module_EXTRA_DEPS
      "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
  endforeach()
else()
  find_package(faiss REQUIRED)
endif()

swig_add_library(swigfaiss
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss.swig
)

if(TARGET swigfaiss AND NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set_target_properties(swigfaiss PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
endif()

set_property(TARGET swigfaiss PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

set_property(SOURCE swigfaiss_avx2.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_avx2)
swig_add_library(swigfaiss_avx2
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_avx2.swig
)
set_property(TARGET swigfaiss_avx2 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(swigfaiss_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigfaiss_avx512.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_avx512)
swig_add_library(swigfaiss_avx512
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_avx512.swig
)
set_property(TARGET swigfaiss_avx512 PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  set_target_properties(swigfaiss_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigfaiss_avx512_spr.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_avx512_spr)
swig_add_library(swigfaiss_avx512_spr
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_avx512_spr.swig
)
set_property(TARGET swigfaiss_avx512_spr PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "avx512_spr")
  set_target_properties(swigfaiss_avx512_spr PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE swigfaiss_sve.swig
  PROPERTY SWIG_MODULE_NAME swigfaiss_sve)
swig_add_library(swigfaiss_sve
  TYPE SHARED
  LANGUAGE python
  SOURCES swigfaiss_sve.swig
)
set_property(TARGET swigfaiss_sve PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
if(NOT FAISS_OPT_LEVEL STREQUAL "sve")
  set_target_properties(swigfaiss_sve PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_property(SOURCE faiss_example_external_module.swig
  PROPERTY SWIG_MODULE_NAME faiss_example_external_module)
swig_add_library(faiss_example_external_module
  TYPE SHARED
  LANGUAGE python
  SOURCES faiss_example_external_module.swig
)
set_property(TARGET faiss_example_external_module PROPERTY SWIG_COMPILE_OPTIONS -doxygen)

if(AIX)
  set_target_properties(swigfaiss PROPERTIES SUFFIX .a)
  set_target_properties(faiss_example_external_module PROPERTIES SUFFIX .a)
elseif(NOT WIN32)
  # PEP 3149 compliant extension suffix, e.g. .cpython-312-darwin.so
  # This ensures the shared object filename matches what Python's import
  # machinery expects for C extension modules.
  if(Python_SOABI)
    set(_FAISS_EXT_SUFFIX ".${Python_SOABI}.so")
  else()
    # Fallback: ask the interpreter for the canonical extension suffix.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
      OUTPUT_VARIABLE _FAISS_EXT_SUFFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _FAISS_EXT_SUFFIX)
      set(_FAISS_EXT_SUFFIX ".so")
    endif()
  endif()
  message(STATUS "Python extension suffix: ${_FAISS_EXT_SUFFIX}")
  set_target_properties(swigfaiss PROPERTIES SUFFIX "${_FAISS_EXT_SUFFIX}")
  set_target_properties(swigfaiss_avx2 PROPERTIES SUFFIX "${_FAISS_EXT_SUFFIX}")
  set_target_properties(swigfaiss_avx512 PROPERTIES SUFFIX "${_FAISS_EXT_SUFFIX}")
  set_target_properties(swigfaiss_avx512_spr PROPERTIES SUFFIX "${_FAISS_EXT_SUFFIX}")
  set_target_properties(swigfaiss_sve PROPERTIES SUFFIX "${_FAISS_EXT_SUFFIX}")
  set_target_properties(faiss_example_external_module PROPERTIES SUFFIX "${_FAISS_EXT_SUFFIX}")
else()
  # we need bigobj for the swig wrapper
  target_compile_options(swigfaiss PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx2 PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx512 PRIVATE /bigobj)
  target_compile_options(swigfaiss_avx512_spr PRIVATE /bigobj)
  target_compile_options(swigfaiss_sve PRIVATE /bigobj)
  target_compile_options(faiss_example_external_module PRIVATE /bigobj)
endif()

if(FAISS_ENABLE_SVS)
  target_compile_definitions(swigfaiss PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_avx2 PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_avx512 PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_avx512_spr PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(swigfaiss_sve PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
  target_compile_definitions(faiss_example_external_module PRIVATE FAISS_ENABLE_SVS FAISS_SVS_RUNTIME_VERSION=${FAISS_SVS_RUNTIME_VERSION})
endif()

# Enable DD support in Python bindings when building with Dynamic Dispatch
if(FAISS_OPT_LEVEL STREQUAL "dd")
  # Add compile definition for the generated C++ wrapper
  target_compile_definitions(swigfaiss PRIVATE FAISS_ENABLE_DD)
endif()

if(FAISS_ENABLE_GPU)
  if(FAISS_ENABLE_ROCM)
    target_link_libraries(swigfaiss PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx2 PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx512 PRIVATE hip::host)
    target_link_libraries(swigfaiss_avx512_spr PRIVATE hip::host)
    target_link_libraries(faiss_example_external_module PRIVATE hip::host)
  else()
    find_package(CUDAToolkit REQUIRED)
    if(FAISS_ENABLE_CUVS AND NOT TARGET cuvs::cuvs)
      find_package(cuvs)
    endif()
    target_link_libraries(swigfaiss PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_avx2 PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_avx512 PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_avx512_spr PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    target_link_libraries(swigfaiss_sve PRIVATE CUDA::cudart $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
  endif()
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swigfaiss PRIVATE
  faiss
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx2 PRIVATE
  faiss_avx2
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx512 PRIVATE
  faiss_avx512
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_avx512_spr PRIVATE
  faiss_avx512_spr
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigfaiss_sve PRIVATE
  faiss_sve
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(faiss_example_external_module PRIVATE
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
  swigfaiss
  faiss
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(swigfaiss PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx512 PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_avx512_spr PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigfaiss_sve PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_example_external_module PRIVATE ${PROJECT_SOURCE_DIR}/../..)

# Note: find_package(Python) was already called at the top of this file;
# the imported targets (Python::Module, Python::NumPy, etc.) and variables
# (Python_INCLUDE_DIRS, Python_SOABI) are already available.

add_library(faiss_python_callbacks OBJECT
  python_callbacks.cpp
)
set_property(TARGET faiss_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

# Hack so that python_callbacks.h can be included as
# `#include <faiss/python/python_callbacks.h>`.
target_include_directories(faiss_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(faiss_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_sources(swigfaiss PRIVATE $<TARGET_OBJECTS:faiss_python_callbacks>)
target_sources(swigfaiss_avx2 PRIVATE $<TARGET_OBJECTS:faiss_python_callbacks>)
target_sources(swigfaiss_avx512 PRIVATE $<TARGET_OBJECTS:faiss_python_callbacks>)
target_sources(swigfaiss_avx512_spr PRIVATE $<TARGET_OBJECTS:faiss_python_callbacks>)
target_sources(swigfaiss_sve PRIVATE $<TARGET_OBJECTS:faiss_python_callbacks>)
target_sources(faiss_example_external_module PRIVATE $<TARGET_OBJECTS:faiss_python_callbacks>)

# ============================================================================
# Legacy file staging (cmake --build && cd build/faiss/python && python setup.py install)
# These configure_file / file(COPY) rules are used by the classic workflow.
# They are harmless when scikit-build-core drives the build.
# ============================================================================
configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)
configure_file(class_wrappers.py class_wrappers.py COPYONLY)
configure_file(gpu_wrappers.py gpu_wrappers.py COPYONLY)
configure_file(extra_wrappers.py extra_wrappers.py COPYONLY)
configure_file(array_conversions.py array_conversions.py COPYONLY)
configure_file(__init__.pyi __init__.pyi COPYONLY)
configure_file(py.typed py.typed COPYONLY)

# file(GLOB files "${PROJECT_SOURCE_DIR}/../../contrib/*.py")
file(COPY ${PROJECT_SOURCE_DIR}/../../contrib  DESTINATION .)

# ============================================================================
# Install rules for scikit-build-core (pip install / uv pip install)
#
# SKBUILD is set by scikit-build-core.  These rules are skipped during
# plain cmake workflows so the existing build/install path is unchanged.
#
# All rules use COMPONENT python so that pyproject.toml's
# install.components = ["python"] keeps headers/cmake-configs out of the wheel.
# ============================================================================
if(SKBUILD)
  set(_FAISS_PKG "faiss")
  set(_FAISS_LIB "faiss/lib")

  # -- Python source files ---------------------------------------------------
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pyi
    ${CMAKE_CURRENT_SOURCE_DIR}/py.typed
    ${CMAKE_CURRENT_SOURCE_DIR}/loader.py
    ${CMAKE_CURRENT_SOURCE_DIR}/class_wrappers.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_wrappers.py
    ${CMAKE_CURRENT_SOURCE_DIR}/extra_wrappers.py
    ${CMAKE_CURRENT_SOURCE_DIR}/array_conversions.py
    DESTINATION ${_FAISS_PKG}
    COMPONENT python
  )

  # -- contrib package -------------------------------------------------------
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../contrib/
    DESTINATION ${_FAISS_PKG}/contrib
    COMPONENT python
    FILES_MATCHING
      PATTERN "*.py"
      PATTERN "__pycache__" EXCLUDE
  )

  # -- SWIG-generated Python wrapper (generic) -------------------------------
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/swigfaiss.py
    DESTINATION ${_FAISS_PKG}
    COMPONENT python
  )

  # -- SWIG native extension (generic) --------------------------------------
  install(TARGETS swigfaiss
    LIBRARY DESTINATION ${_FAISS_PKG} COMPONENT python
    RUNTIME DESTINATION ${_FAISS_PKG} COMPONENT python   # Windows .pyd
  )

  # -- libfaiss shared library (bundled for self-contained wheels) -----------
  # For final wheel repair use delocate-wheel (macOS) or auditwheel (Linux).
  if(TARGET faiss AND BUILD_SHARED_LIBS)
    install(TARGETS faiss
      LIBRARY DESTINATION ${_FAISS_LIB} COMPONENT python
      RUNTIME DESTINATION ${_FAISS_LIB} COMPONENT python
    )
  endif()

  # -- RPATH so _swigfaiss finds libfaiss at runtime -------------------------
  if(APPLE)
    set_target_properties(swigfaiss PROPERTIES INSTALL_RPATH "@loader_path/lib")
  elseif(UNIX)
    set_target_properties(swigfaiss PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
  endif()

  # -- AVX2 variant (built when FAISS_OPT_LEVEL ∈ {avx2, avx512, avx512_spr})
  if(FAISS_OPT_LEVEL STREQUAL "avx2" OR FAISS_OPT_LEVEL STREQUAL "avx512" OR FAISS_OPT_LEVEL STREQUAL "avx512_spr")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/swigfaiss_avx2.py
      DESTINATION ${_FAISS_PKG} COMPONENT python)
    install(TARGETS swigfaiss_avx2
      LIBRARY DESTINATION ${_FAISS_PKG} COMPONENT python
      RUNTIME DESTINATION ${_FAISS_PKG} COMPONENT python)
    if(TARGET faiss_avx2)
      install(TARGETS faiss_avx2
        LIBRARY DESTINATION ${_FAISS_LIB} COMPONENT python
        RUNTIME DESTINATION ${_FAISS_LIB} COMPONENT python)
    endif()
    if(APPLE)
      set_target_properties(swigfaiss_avx2 PROPERTIES INSTALL_RPATH "@loader_path/lib")
    elseif(UNIX)
      set_target_properties(swigfaiss_avx2 PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    endif()
  endif()

  # -- AVX512 variant (built when FAISS_OPT_LEVEL = avx512) -----------------
  if(FAISS_OPT_LEVEL STREQUAL "avx512")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/swigfaiss_avx512.py
      DESTINATION ${_FAISS_PKG} COMPONENT python)
    install(TARGETS swigfaiss_avx512
      LIBRARY DESTINATION ${_FAISS_PKG} COMPONENT python
      RUNTIME DESTINATION ${_FAISS_PKG} COMPONENT python)
    if(TARGET faiss_avx512)
      install(TARGETS faiss_avx512
        LIBRARY DESTINATION ${_FAISS_LIB} COMPONENT python
        RUNTIME DESTINATION ${_FAISS_LIB} COMPONENT python)
    endif()
    if(APPLE)
      set_target_properties(swigfaiss_avx512 PROPERTIES INSTALL_RPATH "@loader_path/lib")
    elseif(UNIX)
      set_target_properties(swigfaiss_avx512 PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    endif()
  endif()

  # -- AVX512-SPR variant (built when FAISS_OPT_LEVEL = avx512_spr) ---------
  if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/swigfaiss_avx512_spr.py
      DESTINATION ${_FAISS_PKG} COMPONENT python)
    install(TARGETS swigfaiss_avx512_spr
      LIBRARY DESTINATION ${_FAISS_PKG} COMPONENT python
      RUNTIME DESTINATION ${_FAISS_PKG} COMPONENT python)
    if(TARGET faiss_avx512_spr)
      install(TARGETS faiss_avx512_spr
        LIBRARY DESTINATION ${_FAISS_LIB} COMPONENT python
        RUNTIME DESTINATION ${_FAISS_LIB} COMPONENT python)
    endif()
    if(APPLE)
      set_target_properties(swigfaiss_avx512_spr PROPERTIES INSTALL_RPATH "@loader_path/lib")
    elseif(UNIX)
      set_target_properties(swigfaiss_avx512_spr PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    endif()
  endif()

  # -- SVE variant (built when FAISS_OPT_LEVEL = sve) -----------------------
  if(FAISS_OPT_LEVEL STREQUAL "sve")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/swigfaiss_sve.py
      DESTINATION ${_FAISS_PKG} COMPONENT python)
    install(TARGETS swigfaiss_sve
      LIBRARY DESTINATION ${_FAISS_PKG} COMPONENT python
      RUNTIME DESTINATION ${_FAISS_PKG} COMPONENT python)
    if(TARGET faiss_sve)
      install(TARGETS faiss_sve
        LIBRARY DESTINATION ${_FAISS_LIB} COMPONENT python
        RUNTIME DESTINATION ${_FAISS_LIB} COMPONENT python)
    endif()
    if(APPLE)
      set_target_properties(swigfaiss_sve PROPERTIES INSTALL_RPATH "@loader_path/lib")
    elseif(UNIX)
      set_target_properties(swigfaiss_sve PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
    endif()
  endif()

endif() # SKBUILD
