#pragma once

#include <QObject>
#include <QString>
#include <QVector>
#include <algorithm>
#include <vector>

#include "analyzer/plugins/analyzerplugin.h"
#include "analyzer/plugins/onnxutil.h"

namespace mixxx {

/// Neural beat tracker "Beat This!" (Foscarin, Schlüter, Widmer; ISMIR 2024,
/// https://github.com/CPJKU/beat_this), run through ONNX Runtime.
///
/// The model sees 30 s chunks of 22050 Hz mono audio and outputs a beat
/// activation at 50 fps. Beats are the activation peaks; no tempo prior and
/// no dynamic Bayesian network are applied, so tempo changes and slow
/// tempos (< 80 BPM) survive. BeatFactory then builds the grid as for the
/// other plugins.
class AnalyzerBeatThis : public AnalyzerBeatsPlugin {
  public:
    static AnalyzerPluginInfo pluginInfo() {
        return AnalyzerPluginInfo(
                "beatthis",
                "CPJKU (Foscarin, Schlüter, Widmer)",
                QObject::tr("Beat This! (neural network)"),
                true);
    }

    /// The model file shipped with Mixxx, below the resource directory.
    static QString modelPath(const QString& resourcePath);

    explicit AnalyzerBeatThis(const QString& modelPath);
    ~AnalyzerBeatThis() override = default;

    AnalyzerPluginInfo info() const override {
        return pluginInfo();
    }

    bool initialize(mixxx::audio::SampleRate sampleRate) override;
    bool processSamples(const CSAMPLE* pIn, SINT iLen) override;
    bool finalize() override;

    bool supportsBeatTracking() const override {
        return true;
    }

    QVector<mixxx::audio::FramePos> getBeats() const override {
        return m_resultBeats;
    }

    // Exposed for tests: the model-independent pre- and post-processing.
    static constexpr int kModelSampleRate = 22050;
    static constexpr int kHop = 441; // 50 frames per second
    static constexpr int kChunkFrames = 1500;
    static constexpr int kBorderFrames = 6;
    static constexpr int kChunkSamples = (kChunkFrames - 1) * kHop;

    /// Chunk start frames, as beat_this.inference.split_piece(avoid_short_end=True).
    static std::vector<int> chunkStarts(int numFrames);
    /// Stitches chunk predictions (first chunk wins in overlaps).
    /// runChunk gets kChunkSamples samples and must fill kChunkFrames logits.
    template<typename RunChunk>
    static std::vector<float> beatLogits(const std::vector<float>& audio, RunChunk runChunk);
    /// Beat times in seconds: local maxima within +-3 frames with logit > 0,
    /// adjacent maxima merged ("minimal" postprocessing of beat_this).
    static std::vector<double> pickBeats(const std::vector<float>& logits);

  private:
    const QString m_modelPath;
    mixxx::audio::SampleRate m_sampleRate;
    MonoResampler m_resampler;
    QVector<mixxx::audio::FramePos> m_resultBeats;
};

template<typename RunChunk>
std::vector<float> AnalyzerBeatThis::beatLogits(
        const std::vector<float>& audio, RunChunk runChunk) {
    const int numFrames = static_cast<int>(audio.size() / kHop) + 1;
    std::vector<float> out(numFrames, -1000.0f);
    std::vector<float> chunk(kChunkSamples);
    std::vector<float> pred(kChunkFrames);
    const std::vector<int> starts = chunkStarts(numFrames);
    // Reverse order + overwrite == earlier chunks win in the overlap.
    for (auto it = starts.rbegin(); it != starts.rend(); ++it) {
        const int start = *it;
        // Frame f is centred on sample f * kHop.
        const long long a0 = static_cast<long long>(start) * kHop;
        std::fill(chunk.begin(), chunk.end(), 0.0f);
        const long long lo = std::max<long long>(a0, 0);
        const long long hi = std::min<long long>(a0 + kChunkSamples,
                static_cast<long long>(audio.size()));
        if (hi > lo) {
            std::copy(audio.begin() + lo, audio.begin() + hi, chunk.begin() + (lo - a0));
        }
        if (!runChunk(chunk, pred)) {
            return {};
        }
        const int f0 = std::max(start + kBorderFrames, 0);
        const int f1 = std::min(start + kChunkFrames - kBorderFrames, numFrames);
        for (int f = f0; f < f1; ++f) {
            out[f] = pred[f - start];
        }
    }
    return out;
}

} // namespace mixxx
