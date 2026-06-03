cmake_minimum_required(VERSION 3.22)

project(ablation_demo LANGUAGES CXX)

find_package(isthmus_cpp CONFIG REQUIRED)

add_executable(ablation_demo
    main.cpp
)

target_link_libraries(ablation_demo PRIVATE isthmus_cpp::isthmus_cpp)

target_compile_features(ablation_demo PRIVATE cxx_std_20)
