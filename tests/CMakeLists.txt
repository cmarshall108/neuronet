# Find the testing framework
find_package(GTest QUIET)

if(NOT GTEST_FOUND)
    # If GTest not found, use FetchContent to download it
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Add the test executable
add_executable(neuronet_tests
    test_main.cpp
    test_tensor.cpp
    test_ops.cpp
    test_nn.cpp
)

# Link with neuronet, GTest, and nlohmann_json
target_link_libraries(neuronet_tests
    PRIVATE
    neuronet
    gtest
    gtest_main
    nlohmann_json::nlohmann_json
)

# Add the test
add_test(NAME neuronet_tests COMMAND neuronet_tests)

# Install tests
install(TARGETS neuronet_tests DESTINATION bin)
