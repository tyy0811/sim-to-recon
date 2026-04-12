cmake_minimum_required(VERSION 3.16)
project(simtorecon_calib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Dependencies ----
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d imgcodecs)

# nlohmann/json (header-only)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# cxxopts (header-only CLI parser)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.1.1
)

FetchContent_MakeAvailable(json cxxopts)

# ---- Library ----
add_library(calib_lib
    src/calib.cpp
)
target_include_directories(calib_lib PUBLIC include)
target_link_libraries(calib_lib PUBLIC
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

# ---- CLI executable ----
add_executable(calib src/main.cpp)
target_link_libraries(calib PRIVATE calib_lib cxxopts::cxxopts)

# ---- Tests ----
enable_testing()
add_subdirectory(tests)
