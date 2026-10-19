#include <dsp/tempotracking/TempoTrackV2.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Detection function frames at 44.1 kHz with a 512 frame step, as used by
// AnalyzerQueenMaryBeats.
constexpr double kFramesPerSecond = 44100.0 / 512.0;
constexpr std::size_t kLength = static_cast<std::size_t>(60 * kFramesPerSecond);
constexpr double kBeat = 60.0 / 128.0 * kFramesPerSecond; // 128 bpm

void addImpulses(std::vector<double>* pDf, double period, double gain) {
    for (double t = 0; t < pDf->size(); t += period) {
        (*pDf)[static_cast<std::size_t>(std::lround(t)) % pDf->size()] += gain;
    }
}

double medianPeriod(std::vector<int> periods) {
    periods.pop_back(); // the last entry is not set by the tempo tracker
    std::sort(periods.begin(), periods.end());
    return periods[periods.size() / 2];
}

std::vector<int> beatPeriod(const std::vector<double>& df) {
    TempoTrackV2 tt(44100, 512);
    std::vector<int> period(df.size() / 128 + 1);
    tt.calculateBeatPeriod(df, period);
    return period;
}

std::vector<int> beatPeriod(const std::vector<double>& df, const std::vector<double>& bassDf) {
    TempoTrackV2 tt(44100, 512);
    std::vector<int> period(df.size() / 128 + 1);
    tt.calculateBeatPeriod(df, bassDf, period);
    return period;
}

TEST(TempoTrackV2Test, BassBandResolvesDottedBeat) {
    // Broadband onsets dominated by a dotted-quarter figure over sixteenths,
    // with weak beats: alone, it is tracked at 1.5 beats (2/3 of the tempo).
    std::vector<double> df(kLength, 0.0);
    addImpulses(&df, 1.5 * kBeat, 1.0);
    addImpulses(&df, kBeat, 0.2);
    addImpulses(&df, kBeat / 4, 0.3);
    // The kick drum repeats at the beat.
    std::vector<double> bassDf(kLength, 0.0);
    addImpulses(&bassDf, kBeat, 1.0);

    EXPECT_NEAR(1.5 * kBeat, medianPeriod(beatPeriod(df)), 2.0);
    EXPECT_NEAR(kBeat, medianPeriod(beatPeriod(df, bassDf)), 2.0);
}

TEST(TempoTrackV2Test, HalfTimeBassKeepsBeat) {
    std::vector<double> df(kLength, 0.0);
    addImpulses(&df, kBeat, 1.0);
    addImpulses(&df, kBeat / 2, 0.5);
    // A half-time bass line must not halve the tempo: the bass band only
    // corrects duple/triple confusions, not the octave.
    std::vector<double> bassDf(kLength, 0.0);
    addImpulses(&bassDf, 2 * kBeat, 1.0);

    const double broadband = medianPeriod(beatPeriod(df));
    EXPECT_NEAR(kBeat, broadband, 2.0);
    EXPECT_EQ(broadband, medianPeriod(beatPeriod(df, bassDf)));
}

TEST(TempoTrackV2Test, SilentBassKeepsBroadbandResult) {
    std::vector<double> df(kLength, 0.0);
    addImpulses(&df, kBeat, 1.0);
    addImpulses(&df, kBeat / 4, 0.3);
    const std::vector<double> bassDf(kLength, 0.0);

    EXPECT_EQ(beatPeriod(df), beatPeriod(df, bassDf));
}

} // namespace
