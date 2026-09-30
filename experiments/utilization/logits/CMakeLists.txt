cmake_minimum_required(VERSION 3.16)
project(hmx_batch_decode_logits LANGUAGES CXX)

set(LLAMA_SOURCE_DIR "" CACHE PATH "Matching llama.cpp-npu source directory")
set(LLAMA_BUILD_DIR "" CACHE PATH "Existing Android HMX build directory")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/include/llama.h" OR
   NOT EXISTS "${LLAMA_BUILD_DIR}/src/libllama.so")
    message(FATAL_ERROR "Provide matching LLAMA_SOURCE_DIR and LLAMA_BUILD_DIR")
endif()
add_library(existing_llama SHARED IMPORTED)
set_target_properties(existing_llama PROPERTIES
    IMPORTED_LOCATION "${LLAMA_BUILD_DIR}/src/libllama.so")

# Resolve these only inside the same supplied build. The ordinary CPU pool
# constructor/free functions and base parameter helpers must match libllama.
foreach(component IN ITEMS ggml-cpu ggml-base)
    set(component_library "")
    foreach(directory IN ITEMS
            "${LLAMA_BUILD_DIR}/ggml/src"
            "${LLAMA_BUILD_DIR}/ggml/src/${component}"
            "${LLAMA_BUILD_DIR}")
        if(EXISTS "${directory}/lib${component}.so")
            set(component_library "${directory}/lib${component}.so")
            break()
        endif()
    endforeach()
    if(NOT component_library)
        message(FATAL_ERROR "Matching lib${component}.so missing from LLAMA_BUILD_DIR")
    endif()
    add_library(existing_${component} SHARED IMPORTED)
    set_target_properties(existing_${component} PROPERTIES
        IMPORTED_LOCATION "${component_library}")
    get_filename_component(component_directory "${component_library}" DIRECTORY)
    list(APPEND matching_library_directories "${component_directory}")
endforeach()
add_executable(hmx-batch-decode-logits batch_decode_logits.cpp)
target_compile_features(hmx-batch-decode-logits PRIVATE cxx_std_17)
target_compile_options(hmx-batch-decode-logits PRIVATE -O3 -Wall -Wextra)
target_include_directories(hmx-batch-decode-logits PRIVATE
    "${LLAMA_SOURCE_DIR}/include" "${LLAMA_SOURCE_DIR}/ggml/include")
target_link_libraries(hmx-batch-decode-logits PRIVATE existing_llama existing_ggml-cpu existing_ggml-base)
target_link_options(hmx-batch-decode-logits PRIVATE
    "-Wl,-rpath-link,${LLAMA_BUILD_DIR}/ggml/src"
    "-Wl,-rpath-link,${LLAMA_BUILD_DIR}/ggml/src/ggml-htp")
foreach(directory IN LISTS matching_library_directories)
    target_link_options(hmx-batch-decode-logits PRIVATE "-Wl,-rpath-link,${directory}")
endforeach()
