set(FAISS_METAL_SRC
  MetalCloner.cpp
  MetalIndex.cpp
  MetalIndexFlat.cpp
  MetalIndexTurboQuant.cpp
  MetalResources.cpp
  StandardMetalResources.mm
  impl/MetalFlatIndex.cpp
  impl/MetalResourcesImpl.mm
  impl/MetalTurboQuantizer.mm
)

set(FAISS_METAL_HEADERS
  MetalCloner.h
  MetalClonerOptions.h
  MetalIndex.h
  MetalIndexFlat.h
  MetalIndexTurboQuant.h
  MetalResources.h
  StandardMetalResources.h
  impl/MetalFlatIndex.h
  impl/MetalKernelsData.h
  impl/MetalResourcesImpl.h
  impl/MetalTurboQuantizer.h
)

set(FAISS_METAL_KERNEL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/FlatSearchKernels.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/TurboQuantKernels.metal
)
set(FAISS_METAL_COMBINE_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/metal_combine_sources.cmake)
set(FAISS_METAL_EMBED_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/metal_embed_metallib.cmake)
string(JOIN ";" FAISS_METAL_KERNEL_INPUTS ${FAISS_METAL_KERNEL_SOURCES})
set(FAISS_METAL_KERNEL ${CMAKE_CURRENT_BINARY_DIR}/MetalKernelsCombined.metal)
set(FAISS_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/MetalKernels.air)
set(FAISS_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/MetalKernels.metallib)
set(FAISS_METAL_DATA_CPP ${CMAKE_CURRENT_BINARY_DIR}/MetalKernelsData.cpp)
set_source_files_properties(${FAISS_METAL_DATA_CPP} PROPERTIES GENERATED TRUE)

add_custom_command(
  OUTPUT ${FAISS_METAL_KERNEL}
  COMMAND ${CMAKE_COMMAND}
    "-DINPUTS=${FAISS_METAL_KERNEL_INPUTS}"
    -DOUTPUT=${FAISS_METAL_KERNEL}
    -P ${FAISS_METAL_COMBINE_SCRIPT}
  DEPENDS ${FAISS_METAL_KERNEL_SOURCES} ${FAISS_METAL_COMBINE_SCRIPT}
  VERBATIM
)

execute_process(
  COMMAND xcrun -sdk macosx metal -v
  RESULT_VARIABLE FAISS_METAL_COMPILER_RESULT
  OUTPUT_QUIET
  ERROR_QUIET)

execute_process(
  COMMAND xcrun -sdk macosx --find metallib
  RESULT_VARIABLE FAISS_METALLIB_RESULT
  OUTPUT_QUIET
  ERROR_QUIET)

if(FAISS_METAL_COMPILER_RESULT EQUAL 0 AND FAISS_METALLIB_RESULT EQUAL 0)
  add_custom_command(
    OUTPUT ${FAISS_METAL_AIR}
    COMMAND xcrun -sdk macosx metal -std=metal3.1 -c ${FAISS_METAL_KERNEL} -o ${FAISS_METAL_AIR}
    DEPENDS ${FAISS_METAL_KERNEL}
    VERBATIM)

  add_custom_command(
    OUTPUT ${FAISS_METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${FAISS_METAL_AIR} -o ${FAISS_METAL_LIB}
    DEPENDS ${FAISS_METAL_AIR}
    VERBATIM)

  add_custom_command(
    OUTPUT ${FAISS_METAL_DATA_CPP}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${FAISS_METAL_LIB} -DOUTPUT=${FAISS_METAL_DATA_CPP} -DMODE=BINARY -P ${FAISS_METAL_EMBED_SCRIPT}
    DEPENDS ${FAISS_METAL_LIB} ${FAISS_METAL_EMBED_SCRIPT}
    VERBATIM)
else()
  message(WARNING "Metal offline compiler toolchain is unavailable; embedding Metal source and using runtime compilation instead.")
  add_custom_command(
    OUTPUT ${FAISS_METAL_DATA_CPP}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${FAISS_METAL_KERNEL} -DOUTPUT=${FAISS_METAL_DATA_CPP} -DMODE=SOURCE -P ${FAISS_METAL_EMBED_SCRIPT}
    DEPENDS ${FAISS_METAL_KERNEL} ${FAISS_METAL_EMBED_SCRIPT}
    VERBATIM)
endif()

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

target_sources(faiss PRIVATE ${FAISS_METAL_SRC} ${FAISS_METAL_DATA_CPP})
target_include_directories(faiss PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_compile_options(faiss PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(faiss PRIVATE
  ${FOUNDATION_FRAMEWORK}
  ${METAL_FRAMEWORK}
  ${MPS_FRAMEWORK})
add_custom_target(faiss_metal_generated DEPENDS ${FAISS_METAL_DATA_CPP})
add_dependencies(faiss faiss_metal_generated)
add_custom_target(faiss_metal DEPENDS faiss)

foreach(header ${FAISS_METAL_HEADERS})
  get_filename_component(dir ${header} DIRECTORY)
  install(FILES ${header}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/metal/${dir})
endforeach()
