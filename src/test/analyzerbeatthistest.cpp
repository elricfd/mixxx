#include <gtest/gtest.h>

#ifdef __BEATTHIS__

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include "analyzer/constants.h"
#include "analyzer/plugins/analyzerbeatthis.h"
#include "test/mixxxtest.h"
#include "track/beatfactory.h"

using mixxx::AnalyzerBeatThis;

namespace {

TEST(AnalyzerBeatThisTest, ChunkStartsMatchBeatThis) {
    // Same as beat_this.inference.split_piece(chunk_size=1500, border_size=6)
    EXPECT_EQ(std::vector<int>({-6}), AnalyzerBeatThis::chunkStarts(100));
    const auto starts = AnalyzerBeatThis::chunkStarts(3000);
    EXPECT_EQ(std::vector<int>({-6, 1482, 3000 - 1494}), starts);
}

TEST(AnalyzerBeatThisTest, PickBeats) {
    std::vector<float> logits(400, -5.0f);
    logits[50] = 2.0f;
    logits[51] = 2.0f; // adjacent equal maxima are merged
    logits[100] = 3.0f;
    logits[150] = 1.0f;
    logits[152] = 0.5f; // not a maximum within +-3 frames
    const auto beats = AnalyzerBeatThis::pickBeats(logits);
    ASSERT_EQ(3u, beats.size());
    EXPECT_DOUBLE_EQ(50.5 / 50, beats[0]);
    EXPECT_DOUBLE_EQ(100.0 / 50, beats[1]);
    EXPECT_DOUBLE_EQ(150.0 / 50, beats[2]);
}

TEST(AnalyzerBeatThisTest, BeatLogitsKeepsFramesAligned) {
    // A fake model that outputs the input sample at each frame centre,
    // i.e. the frame index when the audio is a ramp.
    constexpr int kHop = AnalyzerBeatThis::kHop;
    std::vector<float> audio(AnalyzerBeatThis::kModelSampleRate * 75);
    for (std::size_t i = 0; i < audio.size(); ++i) {
        audio[i] = static_cast<float>(i) / kHop;
    }
    const auto logits = AnalyzerBeatThis::beatLogits(audio,
            [](const std::vector<float>& chunk, std::vector<float>& pred) {
                for (std::size_t f = 0; f < pred.size(); ++f) {
                    pred[f] = f * kHop < chunk.size() ? chunk[f * kHop] : 0.0f;
                }
                return true;
            });
    ASSERT_EQ(audio.size() / kHop + 1, logits.size());
    for (std::size_t f = 0; f + 1 < logits.size(); ++f) {
        ASSERT_FLOAT_EQ(static_cast<float>(f), logits[f]) << "frame " << f;
    }
}

class AnalyzerBeatThisModelTest : public MixxxTest {};

TEST_F(AnalyzerBeatThisModelTest, FindsBeatsOfDrumLoop) {
    const QString modelPath = AnalyzerBeatThis::modelPath(
            getTestDir().filePath(QStringLiteral("../../res")));
    if (!QFileInfo::exists(modelPath)) {
        GTEST_SKIP() << "model not found: " << modelPath.toStdString();
    }
    // 40 s of a 120 BPM rock pattern (kick on every beat, eighth-note
    // hi-hat, noise snare on 2 and 4), stereo 44.1 kHz. Without the snare
    // the model reasonably hears the hi-hat eighths as the beat.
    constexpr int kRate = 44100;
    std::vector<CSAMPLE> stereo(2 * kRate * 40);
    std::mt19937 rng(0);
    std::normal_distribution<double> noise;
    for (int i = 0; i < kRate * 40; ++i) {
        const double t = static_cast<double>(i) / kRate;
        const double beatPhase = std::fmod(t, 0.5);
        const double eighthPhase = std::fmod(t, 0.25);
        const double snarePhase = std::fmod(t + 0.5, 1.0);
        double s = std::exp(-beatPhase * 30) * std::sin(2 * std::numbers::pi * 60 * beatPhase);
        s += 0.3 * std::exp(-eighthPhase * 200) * std::sin(2 * std::numbers::pi * 7000 * t);
        if (snarePhase < 0.12) {
            s += 0.4 * std::exp(-snarePhase * 40) * noise(rng);
        }
        stereo[2 * i] = stereo[2 * i + 1] = static_cast<CSAMPLE>(0.5 * s);
    }
    AnalyzerBeatThis plugin(modelPath);
    ASSERT_TRUE(plugin.initialize(mixxx::audio::SampleRate(kRate)));
    for (std::size_t i = 0; i < stereo.size(); i += 2 * mixxx::kAnalysisFramesPerChunk) {
        const auto len = std::min<std::size_t>(
                2 * mixxx::kAnalysisFramesPerChunk, stereo.size() - i);
        ASSERT_TRUE(plugin.processSamples(&stereo[i], static_cast<SINT>(len)));
    }
    ASSERT_TRUE(plugin.finalize());
    const auto beats = plugin.getBeats();
    ASSERT_GT(beats.size(), 60);
    // Median inter-beat interval: 0.5 s at 120 BPM.
    std::vector<double> ibi;
    for (int i = 1; i < beats.size(); ++i) {
        ibi.push_back((beats[i] - beats[i - 1]) / kRate);
    }
    std::nth_element(ibi.begin(), ibi.begin() + ibi.size() / 2, ibi.end());
    EXPECT_NEAR(0.5, ibi[ibi.size() / 2], 0.021);
}

/// Opt-in check against the Python reference (mik-tempo-research):
/// MIXXX_BEATTHIS_EVAL_LIST names a text file with one raw stereo float32
/// 44.1 kHz file per line; results go to MIXXX_BEATTHIS_EVAL_OUT as CSV
/// (file, beats, bpm, steady coverage), using the same grid code as Mixxx.
TEST_F(AnalyzerBeatThisModelTest, EvalFileList) {
    const QString listPath = qEnvironmentVariable("MIXXX_BEATTHIS_EVAL_LIST");
    const QString outPath = qEnvironmentVariable("MIXXX_BEATTHIS_EVAL_OUT");
    if (listPath.isEmpty() || outPath.isEmpty()) {
        GTEST_SKIP() << "set MIXXX_BEATTHIS_EVAL_LIST and MIXXX_BEATTHIS_EVAL_OUT";
    }
    const QString modelPath = AnalyzerBeatThis::modelPath(
            getTestDir().filePath(QStringLiteral("../../res")));
    QFile list(listPath);
    ASSERT_TRUE(list.open(QIODevice::ReadOnly | QIODevice::Text));
    QFile out(outPath);
    ASSERT_TRUE(out.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream csv(&out);
    csv << "file,beats,bpm,coverage\n";
    constexpr auto kRate = mixxx::audio::SampleRate(44100);
    while (!list.atEnd()) {
        const QString path = QString::fromUtf8(list.readLine()).trimmed();
        if (path.isEmpty()) {
            continue;
        }
        QFile raw(path);
        ASSERT_TRUE(raw.open(QIODevice::ReadOnly)) << path.toStdString();
        const QByteArray bytes = raw.readAll();
        const auto* pSamples = reinterpret_cast<const CSAMPLE*>(bytes.constData());
        const auto numSamples = static_cast<std::size_t>(bytes.size()) / sizeof(CSAMPLE);
        AnalyzerBeatThis plugin(modelPath);
        ASSERT_TRUE(plugin.initialize(kRate));
        for (std::size_t i = 0; i < numSamples; i += 2 * mixxx::kAnalysisFramesPerChunk) {
            const auto len = std::min<std::size_t>(
                    2 * mixxx::kAnalysisFramesPerChunk, numSamples - i);
            ASSERT_TRUE(plugin.processSamples(pSamples + i, static_cast<SINT>(len)));
        }
        ASSERT_TRUE(plugin.finalize());
        const auto beats = plugin.getBeats();
        const auto analyzedFrames = static_cast<mixxx::audio::FrameDiff_t>(numSamples / 2);
        const auto pBeats = beats.size() > 1
                ? BeatFactory::makePreferredBeats(beats, {}, true, kRate, analyzedFrames)
                : nullptr;
        const auto bpm = pBeats
                ? pBeats->getBpmInRange(mixxx::audio::kStartFramePos,
                          mixxx::audio::FramePos(analyzedFrames))
                : mixxx::Bpm();
        csv << '"' << path << "\"," << beats.size() << ','
            << (bpm.isValid() ? bpm.value() : 0.0) << ','
            << (pBeats ? pBeats->getSteadyCoverage().value_or(-1) : -1) << '\n';
    }
}

} // namespace

#endif // __BEATTHIS__
