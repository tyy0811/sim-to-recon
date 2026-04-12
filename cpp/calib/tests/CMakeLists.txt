include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(test_calib test_calib.cpp)
target_link_libraries(test_calib PRIVATE
    calib_lib
    GTest::gtest_main
    ${OpenCV_LIBS}
)

add_test(NAME test_calib COMMAND test_calib)
