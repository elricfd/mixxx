#pragma once

#include <QObject>
#include <QString>

#include "analyzer/plugins/analyzerplugin.h"
#include "analyzer/plugins/onnxutil.h"

namespace mixxx {

/// Neural key detector OpenKeyScan 3 (https://github.com/rekordcloud/openkeyscan-analyzer,
/// MIT; KeyNet after Korzeniowski & Widmer, EUSIPCO 2018), run through ONNX Runtime.
/// The model (CQT front end + CNN, exported by mik-tempo-research/export_openkeyscan_onnx.py)
/// takes the whole track as 44100 Hz mono and returns one global key.
class AnalyzerOpenKeyScan : public AnalyzerKeyPlugin {
  public:
    static AnalyzerPluginInfo pluginInfo() {
        return AnalyzerPluginInfo(
                "openkeyscan",
                "rekordcloud (after Korzeniowski & Widmer)",
                QObject::tr("OpenKeyScan (neural network)"),
                false);
    }

    /// The model file shipped with Mixxx, below the resource directory.
    static QString modelPath(const QString& resourcePath);

    explicit AnalyzerOpenKeyScan(const QString& modelPath);

    AnalyzerPluginInfo info() const override {
        return pluginInfo();
    }

    bool initialize(mixxx::audio::SampleRate sampleRate) override;
    bool processSamples(const CSAMPLE* pIn, SINT iLen) override;
    bool finalize() override;

    KeyChangeList getKeyChanges() const override {
        return m_resultKeys;
    }

    static constexpr int kModelSampleRate = 44100;
    /// Model class to key: Camelot number (index % 12) + 1, minor (A) below 12, major (B) above.
    static mixxx::track::io::key::ChromaticKey keyFromIndex(int index);

  private:
    const QString m_modelPath;
    MonoResampler m_resampler;
    KeyChangeList m_resultKeys;
};

} // namespace mixxx
