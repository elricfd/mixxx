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

// Onset functions of real audio are never silent between onsets.
void addNoise(std::vector<double>* pDf, double gain, unsigned int seed) {
    for (double& value : *pDf) {
        seed = seed * 1664525u + 1013904223u;
        value += gain * (seed >> 8) / static_cast<double>(1 << 24);
    }
}

void addImpulsesAt(std::vector<double>* pDf, double offset, double period, double gain) {
    for (double t = offset; t < pDf->size(); t += period) {
        (*pDf)[static_cast<std::size_t>(std::lround(t)) % pDf->size()] += gain;
    }
}

// Drum and bass two-step: kick on 1 and the "and" of 2, snare on 2 and 4,
// eighth-note hats.
void makeTwoStep(double bpm, std::vector<double>* pDf, std::vector<double>* pBassDf) {
    const double beat = 60.0 / bpm * kFramesPerSecond;
    pDf->assign(kLength, 0.0);
    pBassDf->assign(kLength, 0.0);
    for (std::vector<double>* pOnsets : {pDf, pBassDf}) {
        addImpulsesAt(pOnsets, 0.0, 4 * beat, 1.0);
        addImpulsesAt(pOnsets, 2.5 * beat, 4 * beat, 1.0);
    }
    addImpulsesAt(pDf, beat, 2 * beat, 1.2);
    addImpulses(pDf, beat / 2, 0.3);
    addNoise(pDf, 0.1, 1);
    addNoise(pBassDf, 0.1, 2);
}

double medianInterval(const std::vector<double>& beats) {
    std::vector<double> intervals;
    for (std::size_t i = 1; i < beats.size(); ++i) {
        intervals.push_back(beats[i] - beats[i - 1]);
    }
    std::sort(intervals.begin(), intervals.end());
    return intervals[intervals.size() / 2];
}

std::vector<double> beatsAtPeriod(const std::vector<double>& df,
        const std::vector<double>& bassDf) {
    TempoTrackV2 tt(44100, 512);
    std::vector<double> beats;
    tt.calculateBeats(df, beatPeriod(df, bassDf), beats);
    return beats;
}

std::vector<double> beatsAtLevel(const std::vector<double>& df,
        const std::vector<double>& bassDf) {
    TempoTrackV2 tt(44100, 512);
    std::vector<double> beats;
    tt.calculateBeats(df, bassDf, beats);
    return beats;
}

TEST(TempoTrackV2Test, DoublesHalfTimeDrumAndBass) {
    std::vector<double> df;
    std::vector<double> bassDf;
    makeTwoStep(174.0, &df, &bassDf);
    const double beat = 60.0 / 174.0 * kFramesPerSecond;

    // The comb filter settles on half the tempo; the metrical level is
    // corrected when the beats are tracked.
    EXPECT_NEAR(2 * beat, medianInterval(beatsAtPeriod(df, bassDf)), 2.0);
    EXPECT_NEAR(beat, medianInterval(beatsAtLevel(df, bassDf)), 1.0);
}

TEST(TempoTrackV2Test, KeepsTrackedDrumAndBassBeat) {
    std::vector<double> df;
    std::vector<double> bassDf;
    makeTwoStep(170.0, &df, &bassDf);

    const std::vector<double> beats = beatsAtPeriod(df, bassDf);
    EXPECT_NEAR(60.0 / 170.0 * kFramesPerSecond, medianInterval(beats), 1.0);
    EXPECT_EQ(beats, beatsAtLevel(df, bassDf));
}

TEST(TempoTrackV2Test, KeepsFourOnTheFloorBeat) {
    std::vector<double> df(kLength, 0.0);
    addImpulses(&df, kBeat, 1.0);
    addImpulses(&df, kBeat / 4, 0.3);
    addNoise(&df, 0.1, 1);
    std::vector<double> bassDf(kLength, 0.0);
    addImpulses(&bassDf, kBeat, 1.0);
    addNoise(&bassDf, 0.1, 2);

    const std::vector<double> beats = beatsAtPeriod(df, bassDf);
    EXPECT_NEAR(kBeat, medianInterval(beats), 1.0);
    EXPECT_EQ(beats, beatsAtLevel(df, bassDf));
}

TEST(TempoTrackV2Test, SilentBassKeepsBroadbandResult) {
    std::vector<double> df(kLength, 0.0);
    addImpulses(&df, kBeat, 1.0);
    addImpulses(&df, kBeat / 4, 0.3);
    const std::vector<double> bassDf(kLength, 0.0);

    EXPECT_EQ(beatPeriod(df), beatPeriod(df, bassDf));
}

} // namespace
