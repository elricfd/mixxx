#include <gtest/gtest.h>

#ifdef __OPENKEYSCAN__

#include <QFileInfo>
#include <cmath>
#include <numbers>
#include <vector>

#include "analyzer/constants.h"
#include "analyzer/plugins/analyzeropenkeyscan.h"
#include "test/mixxxtest.h"

using mixxx::AnalyzerOpenKeyScan;
using mixxx::track::io::key::ChromaticKey;

namespace {

TEST(AnalyzerOpenKeyScanTest, KeyFromIndex) {
    // Classes are Camelot 1A..12A, then 1B..12B.
    EXPECT_EQ(ChromaticKey::G_SHARP_MINOR, AnalyzerOpenKeyScan::keyFromIndex(0));  // 1A
    EXPECT_EQ(ChromaticKey::A_MINOR, AnalyzerOpenKeyScan::keyFromIndex(7));        // 8A
    EXPECT_EQ(ChromaticKey::C_SHARP_MINOR, AnalyzerOpenKeyScan::keyFromIndex(11)); // 12A
    EXPECT_EQ(ChromaticKey::B_MAJOR, AnalyzerOpenKeyScan::keyFromIndex(12));       // 1B
    EXPECT_EQ(ChromaticKey::C_MAJOR, AnalyzerOpenKeyScan::keyFromIndex(19));       // 8B
    EXPECT_EQ(ChromaticKey::G_MAJOR, AnalyzerOpenKeyScan::keyFromIndex(20));       // 9B
    EXPECT_EQ(ChromaticKey::E_MAJOR, AnalyzerOpenKeyScan::keyFromIndex(23));       // 12B
    EXPECT_EQ(ChromaticKey::INVALID, AnalyzerOpenKeyScan::keyFromIndex(24));
}

class AnalyzerOpenKeyScanModelTest : public MixxxTest {};

/// 32 s of I-IV-V-I (minor: i-iv-V-i) with harmonic-rich tones, stereo 44.1 kHz.
std::vector<CSAMPLE> cadence(int rootFromC, bool minor) {
    constexpr int kRate = 44100;
    const int third = minor ? 3 : 4;
    const int chords[4][3] = {{0, third, 7}, {5, 5 + third, 12}, {7, 11, 14}, {0, third, 7}};
    std::vector<CSAMPLE> stereo;
    for (int rep = 0; rep < 4; ++rep) {
        for (const auto& chord : chords) {
            for (int i = 0; i < 2 * kRate; ++i) {
                const double t = static_cast<double>(i) / kRate;
                double s = 0;
                for (int semi : {chord[0], chord[1], chord[2], chord[0] - 12}) {
                    const double f = 261.63 * std::pow(2.0, (rootFromC + semi) / 12.0);
                    for (int k = 1; k <= 5; ++k) {
                        s += std::sin(2 * std::numbers::pi * f * k * t) / k;
                    }
                }
                const auto sample = static_cast<CSAMPLE>(0.05 * s * std::exp(-t * 0.8));
                stereo.push_back(sample);
                stereo.push_back(sample);
            }
        }
    }
    return stereo;
}

TEST_F(AnalyzerOpenKeyScanModelTest, FindsKeyOfCadence) {
    const QString modelPath = AnalyzerOpenKeyScan::modelPath(
            getTestDir().filePath(QStringLiteral("../../res")));
    if (!QFileInfo::exists(modelPath)) {
        GTEST_SKIP() << "model not found: " << modelPath.toStdString();
    }
    const struct {
        int root;
        bool minor;
        ChromaticKey key;
    } cases[] = {{0, false, ChromaticKey::C_MAJOR},
            {9, true, ChromaticKey::A_MINOR},
            {7, false, ChromaticKey::G_MAJOR},
            {2, true, ChromaticKey::D_MINOR},
            {3, false, ChromaticKey::E_FLAT_MAJOR},
            {6, true, ChromaticKey::F_SHARP_MINOR}};
    for (const auto& c : cases) {
        const std::vector<CSAMPLE> stereo = cadence(c.root, c.minor);
        AnalyzerOpenKeyScan plugin(modelPath);
        ASSERT_TRUE(plugin.initialize(mixxx::audio::SampleRate(44100)));
        for (std::size_t i = 0; i < stereo.size(); i += 2 * mixxx::kAnalysisFramesPerChunk) {
            const auto len = std::min<std::size_t>(
                    2 * mixxx::kAnalysisFramesPerChunk, stereo.size() - i);
            ASSERT_TRUE(plugin.processSamples(&stereo[i], static_cast<SINT>(len)));
        }
        ASSERT_TRUE(plugin.finalize());
        const KeyChangeList keys = plugin.getKeyChanges();
        ASSERT_EQ(1, keys.size());
        EXPECT_EQ(c.key, keys.first().first) << "root " << c.root << " minor " << c.minor;
    }
}

TEST_F(AnalyzerOpenKeyScanModelTest, MissingModelFailsInitialize) {
    AnalyzerOpenKeyScan plugin(QStringLiteral("does/not/exist.onnx"));
    EXPECT_FALSE(plugin.initialize(mixxx::audio::SampleRate(44100)));
}

} // namespace

#endif // __OPENKEYSCAN__
