# Example applications
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE neuronet nlohmann_json::nlohmann_json)

add_executable(huggingface_example huggingface_example.cpp)
target_link_libraries(huggingface_example PRIVATE neuronet nlohmann_json::nlohmann_json)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE neuronet nlohmann_json::nlohmann_json)

# Install examples
install(TARGETS 
    simple_example 
    huggingface_example
    benchmark
    DESTINATION bin)
