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

TEST(BeatUtilsTest, TempoMapListsTempoChanges) {
    // 60 s at 120 BPM, then 60 s at 140 BPM.
    auto beats = clickBeats(120.0, 121);
    const double beatLength140 = 60.0 * kSampleRate.value() / 140.0;
    for (int i = 1; i <= 140; ++i) {
        beats.append(beats[120] + i * beatLength140);
    }
    const auto regions = BeatUtils::retrieveConstRegions(beats, kSampleRate);
    EXPECT_EQ(QStringLiteral("0:00 120.00, 1:00 140.00"), BeatUtils::tempoMap(regions, kSampleRate));
    // One tempo only: nothing beyond the BPM to report.
    EXPECT_TRUE(BeatUtils::tempoMap(
            BeatUtils::retrieveConstRegions(clickBeats(128.0, 200), kSampleRate), kSampleRate)
                    .isEmpty());
}

} // namespace
