#include "analyzer/plugins/analyzerqueenmarybeats.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "util/math.h"

namespace {

constexpr int kSampleRate = 44100;
constexpr double kBpm = 128.0;

// Stereo loop: decaying 50-110 Hz kick on the beats and short noise hi-hats
// on the sixteenths.
std::vector<CSAMPLE> makeLoop(double seconds) {
    const int frames = static_cast<int>(seconds * kSampleRate);
    const double beatFrames = 60.0 / kBpm * kSampleRate;
    std::vector<CSAMPLE> samples(2 * frames, 0.0f);
    unsigned int noise = 12345;
    for (int i = 0; i < frames; ++i) {
        const double inBeat = std::fmod(i, beatFrames) / kSampleRate;
        const double inSixteenth = std::fmod(i, beatFrames / 4) / kSampleRate;
        noise = noise * 1664525u + 1013904223u;
        const double white = static_cast<double>(noise >> 8) / (1 << 24) - 0.5;
        const double kick = std::sin(2 * M_PI * (50 + 60 * std::exp(-inBeat * 30)) * inBeat) *
                std::exp(-inBeat * 12);
        const double hat = 0.3 * white * std::exp(-inSixteenth * 150);
        samples[2 * i] = samples[2 * i + 1] = static_cast<CSAMPLE>(0.5 * (kick + hat));
    }
    return samples;
}

TEST(AnalyzerQueenMaryBeatsTest, ConstantTempoLoop) {
    mixxx::AnalyzerQueenMaryBeats analyzer;
    ASSERT_TRUE(analyzer.initialize(mixxx::audio::SampleRate(kSampleRate)));
    const std::vector<CSAMPLE> samples = makeLoop(30.0);
    constexpr std::size_t kChunk = 2 * 4096;
    for (std::size_t i = 0; i < samples.size(); i += kChunk) {
        const auto len = static_cast<SINT>(std::min(kChunk, samples.size() - i));
        ASSERT_TRUE(analyzer.processSamples(samples.data() + i, len));
    }
    ASSERT_TRUE(analyzer.finalize());

    const QVector<mixxx::audio::FramePos> beats = analyzer.getBeats();
    ASSERT_GT(beats.size(), 50);
    std::vector<double> intervals;
    for (int i = 1; i < beats.size(); ++i) {
        intervals.push_back(beats[i].value() - beats[i - 1].value());
    }
    std::nth_element(intervals.begin(), intervals.begin() + intervals.size() / 2, intervals.end());
    const double bpm = 60.0 * kSampleRate / intervals[intervals.size() / 2];
    EXPECT_NEAR(kBpm, bpm, 1.5);
}

} // namespace
