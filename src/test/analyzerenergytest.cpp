#include <gtest/gtest.h>

#include <QFile>
#include <QTextStream>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#ifdef __ENERGY__
#include "analyzer/analyzerenergy.h"

namespace {

using mixxx::energy::FeatureAccumulator;
using mixxx::energy::Features;
using mixxx::energy::kSampleRate;

std::optional<Features> featuresOf(const std::vector<float>& mono) {
    FeatureAccumulator acc;
    acc.addSamples(mono.data(), mono.size());
    return acc.features();
}

TEST(AnalyzerEnergyTest, TreesMatchPython) {
    // Feature rows and predictions from mik-tempo-research/cache/energy_parity.csv
    const std::vector<std::pair<Features, int>> rows = {
            {{-14.879704475402832, -10.338704109191895, 5.264172077178955, 409.81427001953125, 0.20943742187007744, 0.8140127062797546, 0.01317679788917303}, 4},
            {{-55.011077880859375, -50.836936950683594, 2.918975830078125, 251.1735076904297, 4.8222009200899505, 0.00664371345192194, 0.6280497312545776}, 3},
            {{-12.425468444824219, -5.172194480895996, 8.177484512329102, 377.67901611328125, 0.746336987816119, 0.2459114044904709, 0.031181206926703453}, 5},
            {{-7.871667861938477, -4.852119445800781, 3.558318614959717, 460.7655029296875, 0.58650321703722, 0.5892805457115173, 0.04627207666635513}, 7},
            {{-13.265052795410156, -8.715691566467285, 6.034651279449463, 294.32098388671875, 0.5046908020493472, 0.14223027229309082, 0.006296700797975063}, 2},
            {{-5.436132431030273, -3.9568097591400146, 2.2024753093719482, 363.0706481933594, 0.4435074450701484, 0.7647545337677002, 0.03571658954024315}, 6},
    };
    for (const auto& [features, energy] : rows) {
        EXPECT_EQ(energy, mixxx::energy::predictEnergy(features));
    }
}

TEST(AnalyzerEnergyTest, FeaturesFollowTheSignal) {
    std::mt19937 rng(0);
    std::normal_distribution<float> noise;
    std::vector<float> quiet(kSampleRate * 5);
    std::vector<float> loud(kSampleRate * 5);
    std::vector<float> tone(kSampleRate * 5);
    for (std::size_t i = 0; i < quiet.size(); ++i) {
        const float n = noise(rng);
        quiet[i] = 0.01f * n;
        loud[i] = 0.5f * n;
        tone[i] = static_cast<float>(std::sin(2 * std::numbers::pi * 60 * i / kSampleRate));
    }
    const auto q = featuresOf(quiet);
    const auto l = featuresOf(loud);
    ASSERT_TRUE(q && l);
    EXPECT_GT((*l)[0] - (*q)[0], 30);       // louder
    EXPECT_NEAR((*l)[4], (*q)[4], 0.5);     // same brightness
    EXPECT_GT((*featuresOf(tone))[5], 0.9); // 60 Hz is all below 150 Hz
    EXPECT_FALSE(featuresOf(std::vector<float>(kSampleRate * 5, 0.0f)));
}

/// Opt-in: MIXXX_ENERGY_EVAL_LIST names a file of paths to raw mono float32 audio at
/// 22050 Hz; MIXXX_ENERGY_EVAL_OUT gets "path,features...,energy" to compare with energy_fit.py.
TEST(AnalyzerEnergyTest, EvalFileList) {
    const QString listPath = qEnvironmentVariable("MIXXX_ENERGY_EVAL_LIST");
    const QString outPath = qEnvironmentVariable("MIXXX_ENERGY_EVAL_OUT");
    if (listPath.isEmpty() || outPath.isEmpty()) {
        GTEST_SKIP() << "set MIXXX_ENERGY_EVAL_LIST and MIXXX_ENERGY_EVAL_OUT";
    }
    QFile list(listPath);
    ASSERT_TRUE(list.open(QIODevice::ReadOnly | QIODevice::Text));
    QFile out(outPath);
    ASSERT_TRUE(out.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream csv(&out);
    csv.setRealNumberPrecision(10);
    while (!list.atEnd()) {
        const QString path = QString::fromUtf8(list.readLine()).trimmed();
        if (path.isEmpty()) {
            continue;
        }
        QFile raw(path);
        ASSERT_TRUE(raw.open(QIODevice::ReadOnly));
        const QByteArray bytes = raw.readAll();
        const auto* pSamples = reinterpret_cast<const float*>(bytes.constData());
        const auto features = featuresOf(std::vector<float>(pSamples, pSamples + bytes.size() / sizeof(float)));
        csv << '"' << path << '"';
        if (features) {
            for (double f : *features) {
                csv << ',' << f;
            }
            csv << ',' << mixxx::energy::predictEnergy(*features);
        }
        csv << '\n';
    }
}

} // namespace

#endif // __ENERGY__
