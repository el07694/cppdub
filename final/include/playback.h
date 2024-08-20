#ifndef PLAYBACK_H
#define PLAYBACK_H

#include <string>
#include "utils.h"
#include <vector>
#include "portaudio.h"
namespace cppdub {

void _play_with_ffplay( AudioSegment& audio_segment);
bool _play_with_portaudio_safe(AudioSegment& audio_segment);
void play(AudioSegment& audio_segment);

}
#endif // PLAYBACK_H
