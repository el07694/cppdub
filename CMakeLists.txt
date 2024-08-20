#cmake command: cmake -S . -B build -G "Visual Studio 17 2022" -A x64

cmake_minimum_required(VERSION 3.10)

# Define the project name and version
project(cppdub VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/lib/ffmpeg/include
    ${PROJECT_SOURCE_DIR}/lib/portaudio/include
	${PROJECT_SOURCE_DIR}/lib/json-develop/include
)

# Define the source files
set(SOURCES
    src/audio_segment.cpp
    src/cppaudioop.cpp
    src/effects.cpp
    src/exceptions.cpp
    src/generators.cpp
    src/logging_utils.cpp
    src/playback.cpp
    src/silence.cpp
    src/utils.cpp
)

# Create the library
add_library(cppdub STATIC ${SOURCES})

# Define the libraries to link against
find_library(FFMPEG_LIBRARIES
    NAMES avcodec avdevice avfilter avformat avutil postproc swresample swscale
    PATHS ${PROJECT_SOURCE_DIR}/lib/ffmpeg/lib
)

find_library(PORTAUDIO_LIBRARIES
    NAMES portaudio
    PATHS ${PROJECT_SOURCE_DIR}/lib/portaudio/lib
)

# Link the libraries to the cppdub library
target_link_libraries(cppdub
    PRIVATE
    ${FFMPEG_LIBRARIES}
    ${PORTAUDIO_LIBRARIES}
)

# Specify the output directory for the built library
set_target_properties(cppdub PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Optionally: Create an install target
install(TARGETS cppdub DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Optionally: Add a target for testing or examples
# add_subdirectory(tests)
# add_subdirectory(examples)
