#include <gtest/gtest.h>

#include <QVector>

#include "track/beatutils.h"

namespace {

constexpr auto kSampleRate = mixxx::audio::SampleRate(44100);

QVector<mixxx::audio::FramePos> clickBeats(double bpm, int count) {
    QVector<mixxx::audio::FramePos> beats;
    const double beatLength = 60.0 * kSampleRate.value() / bpm;
    for (int i = 0; i < count; ++i) {
        beats.append(mixxx::audio::FramePos(1000 + i * beatLength));
    }
    return beats;
}

mixxx::Bpm constBpm(double bpm) {
    const auto regions = BeatUtils::retrieveConstRegions(clickBeats(bpm, 200), kSampleRate);
    auto firstBeat = mixxx::audio::kStartFramePos;
    return BeatUtils::makeConstBpm(regions, kSampleRate, &firstBeat);
}

TEST(BeatUtilsTest, MakeConstBpmKeepsOrdinaryTempo) {
    EXPECT_DOUBLE_EQ(128.0, constBpm(128.0).value());
    EXPECT_DOUBLE_EQ(70.0, constBpm(70.0).value());
    EXPECT_DOUBLE_EQ(200.0, constBpm(200.0).value());
}

TEST(BeatUtilsTest, SteadyCoverage) {
    const auto regions = BeatUtils::retrieveConstRegions(clickBeats(120.0, 41), kSampleRate);
    // 40 beats at 120 BPM = 20 s steady out of 40 s analyzed.
    const auto coverage = BeatUtils::steadyCoverage(regions, 40.0 * kSampleRate.value());
    ASSERT_TRUE(coverage);
    EXPECT_NEAR(0.5, *coverage, 1e-6);
    EXPECT_FALSE(BeatUtils::steadyCoverage(regions, 0));
}

} // namespace
