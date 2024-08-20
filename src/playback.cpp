#include "playback.h"
#include <cstdlib>      // For system()
#include <cstdio>       // For tmpnam()
#include <fstream>      // For file operations
#include <iostream>     // For error handling
#include <vector>
#include <map>
#include "audio_segment.h"

namespace cppdub {

void _play_with_ffplay(AudioSegment& audio_segment) {
    // Get the player name (e.g., "ffplay")
    std::string PLAYER = get_player_name();

    // Create a temporary file for the WAV output
    char temp_file_name[L_tmpnam];
    std::tmpnam(temp_file_name);  // Generate temporary file name

    // Export the audio segment to the temporary file in WAV format
    const std::string format = "wav";
    const std::string codec;
    const std::string bitrate;
    const std::vector<std::string> parameters;
    const std::map<std::string, std::string> tags;
    const std::string id3v2_version;
    const std::string cover;

    // Attempt to export the audio segment to the temporary file
    std::ofstream out_file = audio_segment.export_segment(static_cast<std::string>(temp_file_name), format, codec, bitrate, parameters, tags, id3v2_version, cover);


    // Construct the ffplay command
    std::string command = PLAYER + " -nodisp -autoexit -hide_banner " + std::string(temp_file_name);

    // Play the audio using ffplay
    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "Failed to play audio with ffplay." << std::endl;
    }

    // Clean up the temporary file if needed
    std::remove(temp_file_name);  // Deletes the temporary file
}

// The primary function for playing audio using PortAudio
bool _play_with_portaudio_safe(AudioSegment& audio_segment) {
    PaStream* stream;
    PaError err;

    // Initialize PortAudio
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return false;  // Mimic Python's ImportError by returning failure
    }

    // Open the audio stream
    err = Pa_OpenDefaultStream(&stream,
                               0, // No input channels
                               audio_segment.get_channels(), // Output channels
                               paInt16, // Assuming 16-bit audio; adapt based on `audio_segment.sample_width`
                               audio_segment.get_frame_rate(), // Frame rate
                               256, // Frames per buffer; choose a size that works for you
                               NULL, // No callback, we'll use blocking API
                               NULL); // No callback data
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return false;
    }

    // Start the audio stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return false;
    }

    try {
        // Stream the audio chunks
        int chunk_time = 500; // 500 milliseconds
        int total_duration = static_cast<int>(audio_segment.length_in_milliseconds());
        int start = 0;
        int end = 0;
        while (end < total_duration) {
            start = end;
            if (start > total_duration) {
                break;
            }
            if (start + chunk_time < total_duration) {
                end = start + chunk_time;
            } else {
                end = total_duration;
            }

            // Extract the chunk
            AudioSegment chunk = audio_segment.slice(static_cast<int64_t>(start), static_cast<int64_t>(end));  // Assuming slice function

            // Write the chunk to the PortAudio stream
            const int16_t* raw_data = reinterpret_cast<const int16_t*>(chunk.raw_data().data());
            size_t total_samples = chunk.raw_data().size() / sizeof(int16_t); // Total number of samples
            unsigned long frames = total_samples / audio_segment.get_channels();   // Number of frames

            err = Pa_WriteStream(stream, raw_data, frames);
            if (err != paNoError) {
                std::cerr << "PortAudio error while writing to stream: " << Pa_GetErrorText(err) << std::endl;
                break;
            }
        }

    } catch (...) {
        std::cerr << "Exception occurred during playback." << std::endl;
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return false;  // Handle any exception as a failure
    }

    // Stop and close the audio stream
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    }

    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
    }

    // Terminate PortAudio
    Pa_Terminate();

    return true;  // Successfully played the audio
}

void play(AudioSegment& audio_segment) {
    // Try to play with PortAudio first
    bool played_with_portaudio = _play_with_portaudio_safe(audio_segment);

    // If PortAudio playback failed, fall back to ffplay
    if (!played_with_portaudio) {
        _play_with_ffplay(audio_segment);
    }
}

}