cmake_minimum_required(VERSION 3.0)
project(resnet_intel_onednn)

set(CMAKE_CXX_STANDARD 17)

set(DNNLROOT /opt/intel/oneapi/2024.0/)

include_directories(${DNNLROOT}/include)
link_directories(${DNNLROOT}/lib)

add_compile_options(-fsycl)
add_link_options(-tbb -dnnl)

add_executable(getting_started getting_started.cpp)
add_executable(resnet resnet.cpp)

target_link_libraries(getting_started PRIVATE dnnl sycl)
target_link_libraries(cnn_training PRIVATE dnnl sycl)
