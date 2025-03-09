# Example applications
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PRIVATE neuronet)

add_executable(huggingface_example huggingface_example.cpp)
target_link_libraries(huggingface_example PRIVATE neuronet)

# Install examples
install(TARGETS 
    simple_example 
    huggingface_example
    DESTINATION bin)
