include(GoogleTest)

add_library(faiss_metal_test_helper INTERFACE)
target_link_libraries(faiss_metal_test_helper INTERFACE faiss GTest::gtest_main)
target_compile_options(faiss_metal_test_helper INTERFACE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
if(APPLE)
  find_library(OBJC_LIBRARY objc REQUIRED)
  target_link_libraries(faiss_metal_test_helper INTERFACE ${OBJC_LIBRARY})
endif()

set(FAISS_METAL_TEST_TARGETS)

macro(faiss_add_metal_test file)
  get_filename_component(test_name ${file} NAME_WE)
  add_executable(${test_name} ${file})
  target_link_libraries(${test_name} PRIVATE faiss_metal_test_helper)
  target_compile_options(${test_name} PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  gtest_discover_tests(${test_name})
  list(APPEND FAISS_METAL_TEST_TARGETS ${test_name})
endmacro()

faiss_add_metal_test(TestMetalIndexFlat.mm)
faiss_add_metal_test(TestMetalTurboQuant.cpp)
faiss_add_metal_test(TestMetalCloner.cpp)

add_custom_target(faiss_metal_test DEPENDS ${FAISS_METAL_TEST_TARGETS})
