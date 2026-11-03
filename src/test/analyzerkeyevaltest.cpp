#include <gtest/gtest.h>

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <memory>

#include "analyzer/constants.h"
#include "analyzer/plugins/analyzerqueenmarykey.h"
#ifdef __KEYFINDER__
#include "analyzer/plugins/analyzerkeyfinder.h"
#endif
#ifdef __OPENKEYSCAN__
#include "analyzer/plugins/analyzeropenkeyscan.h"
#endif
#include "test/mixxxtest.h"
#include "track/keyfactory.h"
#include "track/keyutils.h"

namespace {

/// Opt-in comparison of the key detectors on real audio (mik-tempo-research/key_eval.py):
/// MIXXX_KEY_EVAL_LIST names a text file with "rate<TAB>path" per line, each path a raw
/// interleaved stereo float32 file at that rate; results go to MIXXX_KEY_EVAL_OUT as CSV
/// (file, plugin, global key in Lancelot/Camelot notation) using the library's global-key rule.
class AnalyzerKeyEvalTest : public MixxxTest {};

TEST_F(AnalyzerKeyEvalTest, EvalFileList) {
    const QString listPath = qEnvironmentVariable("MIXXX_KEY_EVAL_LIST");
    const QString outPath = qEnvironmentVariable("MIXXX_KEY_EVAL_OUT");
    if (listPath.isEmpty() || outPath.isEmpty()) {
        GTEST_SKIP() << "set MIXXX_KEY_EVAL_LIST and MIXXX_KEY_EVAL_OUT";
    }
    QFile list(listPath);
    ASSERT_TRUE(list.open(QIODevice::ReadOnly | QIODevice::Text));
    QFile out(outPath);
    ASSERT_TRUE(out.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream csv(&out);
    csv << "file,plugin,key\n";
    while (!list.atEnd()) {
        const QStringList line = QString::fromUtf8(list.readLine()).trimmed().split('\t');
        if (line.size() != 2) {
            continue;
        }
        const auto rate = mixxx::audio::SampleRate(line[0].toUInt());
        QFile raw(line[1]);
        ASSERT_TRUE(raw.open(QIODevice::ReadOnly)) << line[1].toStdString();
        const QByteArray bytes = raw.readAll();
        const auto* pSamples = reinterpret_cast<const CSAMPLE*>(bytes.constData());
        const auto numSamples = static_cast<std::size_t>(bytes.size()) / sizeof(CSAMPLE);
        std::vector<std::unique_ptr<mixxx::AnalyzerKeyPlugin>> plugins;
        plugins.push_back(std::make_unique<mixxx::AnalyzerQueenMaryKey>());
#ifdef __KEYFINDER__
        plugins.push_back(std::make_unique<mixxx::AnalyzerKeyFinder>());
#endif
#ifdef __OPENKEYSCAN__
        const QString keyModel = mixxx::AnalyzerOpenKeyScan::modelPath(
                getTestDir().filePath(QStringLiteral("../../res")));
        if (QFileInfo::exists(keyModel)) {
            plugins.push_back(std::make_unique<mixxx::AnalyzerOpenKeyScan>(keyModel));
        }
#endif
        for (const auto& pPlugin : plugins) {
            ASSERT_TRUE(pPlugin->initialize(rate));
            for (std::size_t i = 0; i < numSamples; i += 2 * mixxx::kAnalysisFramesPerChunk) {
                const auto len = std::min<std::size_t>(
                        2 * mixxx::kAnalysisFramesPerChunk, numSamples - i);
                ASSERT_TRUE(pPlugin->processSamples(pSamples + i, static_cast<SINT>(len)));
            }
            ASSERT_TRUE(pPlugin->finalize());
            const Keys keys = KeyFactory::makePreferredKeys(pPlugin->getKeyChanges(),
                    {},
                    rate,
                    static_cast<SINT>(numSamples / 2));
            csv << '"' << line[1] << "\"," << pPlugin->info().id() << ','
                << KeyUtils::keyToString(keys.getGlobalKey(), KeyUtils::KeyNotation::Lancelot)
                << '\n';
        }
    }
}

} // namespace
