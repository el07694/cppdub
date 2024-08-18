cmake_minimum_required(VERSION 3.10)
project(CppDub)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories for headers
include_directories(cpp/include)
include_directories(lib/json-develop/include)
include_directories(lib/ffmpeg-7.0.2/include)
include_directories(lib/portaudio/include)

# Source files
set(SOURCES
    cpp/src/audio_segment.cpp
    cpp/src/cppaudioop.cpp
    cpp/src/effects.cpp
    cpp/src/exceptions.cpp
    cpp/src/generators.cpp
    cpp/src/logging_utils.cpp
    cpp/src/playback.cpp
    cpp/src/silence.cpp
    cpp/src/utils.cpp
)

# Header files (optional, just for clarity, not mandatory to list them)
set(HEADERS
    cpp/include/audio_segment.h
    cpp/include/cppaudioop.h
    cpp/include/effects.h
    cpp/include/exceptions.h
    cpp/include/generators.h
    cpp/include/logging_utils.h
    cpp/include/playback.h
    cpp/include/silence.h
    cpp/include/utils.h
)

# Add executable or library (uncomment one of them based on your project)
# add_executable(CppDub ${SOURCES} ${HEADERS})
add_library(CppDub STATIC ${SOURCES} ${HEADERS})

# Link FFmpeg libraries (adjust the names as necessary based on your build)
find_library(FFMPEG_AVCODEC avcodec PATHS lib/ffmpeg-7.0.2/lib)
find_library(FFMPEG_AVFORMAT avformat PATHS lib/ffmpeg-7.0.2/lib)
find_library(FFMPEG_AVUTIL avutil PATHS lib/ffmpeg-7.0.2/lib)

# Link PortAudio library
find_library(PORTAUDIO_LIB portaudio PATHS lib/portaudio/lib)

# Link JSON if it is built as a library (some versions are header-only)
# You may skip this if json-develop is header-only
# find_library(JSON_LIB nlohmann_json PATHS lib/json-develop/lib)

# Add libraries to the project
target_link_libraries(CppDub ${FFMPEG_AVCODEC} ${FFMPEG_AVFORMAT} ${FFMPEG_AVUTIL} ${PORTAUDIO_LIB})

# Installation setup
install(TARGETS CppDub
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install header files
install(FILES ${HEADERS} DESTINATION include/CppDub)

# Optional: Add custom logic for installing external dependencies if necessary
