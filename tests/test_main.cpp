#include <gtest/gtest.h>
#include <neuronet/neuronet.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize the NeuroNet library
    neuronet::initialize();
    
    // Run all tests
    int result = RUN_ALL_TESTS();
    
    // Clean up
    neuronet::cleanup();
    
    return result;
}
