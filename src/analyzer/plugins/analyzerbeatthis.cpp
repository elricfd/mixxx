#include "analyzer/plugins/analyzerbeatthis.h"

#include <onnxruntime_cxx_api.h>

#include <QDir>
#include <QFileInfo>
#include <QtDebug>
#include <array>

#include "analyzer/constants.h"

namespace mixxx {

// static
QString AnalyzerBeatThis::modelPath(const QString& resourcePath) {
    return QDir(resourcePath).filePath(QStringLiteral("models/beatthis_final0.onnx"));
}

AnalyzerBeatThis::AnalyzerBeatThis(const QString& modelPath)
        : m_modelPath(modelPath),
          m_resampler(kModelSampleRate) {
}

// static
std::vector<int> AnalyzerBeatThis::chunkStarts(int numFrames) {
    std::vector<int> starts;
    for (int s = -kBorderFrames; s < numFrames - kBorderFrames;
            s += kChunkFrames - 2 * kBorderFrames) {
        starts.push_back(s);
    }
    if (numFrames > kChunkFrames - 2 * kBorderFrames) {
        // Shift the last chunk left so it ends at the end of the piece.
        starts.back() = numFrames - (kChunkFrames - kBorderFrames);
    }
    return starts;
}

// static
std::vector<double> AnalyzerBeatThis::pickBeats(const std::vector<float>& logits) {
    constexpr int kRadius = 3; // +-70 ms at 50 fps
    const int n = static_cast<int>(logits.size());
    std::vector<double> beats;
    double groupSum = 0;
    int groupSize = 0;
    int lastFrame = -2;
    for (int f = 0; f < n; ++f) {
        if (logits[f] <= 0) {
            continue;
        }
        bool isMax = true;
        for (int g = std::max(0, f - kRadius); g <= std::min(n - 1, f + kRadius); ++g) {
            if (logits[g] > logits[f]) {
                isMax = false;
                break;
            }
        }
        if (!isMax) {
            continue;
        }
        if (groupSize > 0 && f - lastFrame > 1) {
            beats.push_back(groupSum / groupSize / (kModelSampleRate / kHop));
            groupSum = 0;
            groupSize = 0;
        }
        groupSum += f;
        ++groupSize;
        lastFrame = f;
    }
    if (groupSize > 0) {
        beats.push_back(groupSum / groupSize / (kModelSampleRate / kHop));
    }
    return beats;
}

bool AnalyzerBeatThis::initialize(mixxx::audio::SampleRate sampleRate) {
    m_resultBeats.clear();
    m_sampleRate = sampleRate;
    if (!QFileInfo::exists(m_modelPath)) {
        qWarning() << "Beat This!: model file not found:" << m_modelPath;
        return false;
    }
    return m_resampler.initialize(sampleRate);
}

bool AnalyzerBeatThis::processSamples(const CSAMPLE* pIn, SINT iLen) {
    return m_resampler.process(pIn, iLen);
}

bool AnalyzerBeatThis::finalize() {
    if (!m_resampler.finish()) {
        return false;
    }
    const std::shared_ptr<Ort::Session> pSession = onnxSession(m_modelPath);
    if (!pSession) {
        return false;
    }
    const Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> inputShape{1, kChunkSamples};
    const std::array<int64_t, 2> outputShape{1, kChunkFrames};
    const char* inputNames[] = {"audio"};
    const char* outputNames[] = {"beat"};
    auto runChunk = [&](std::vector<float>& chunk, std::vector<float>& pred) {
        try {
            Ort::Value input = Ort::Value::CreateTensor<float>(memoryInfo,
                    chunk.data(),
                    chunk.size(),
                    inputShape.data(),
                    inputShape.size());
            Ort::Value output = Ort::Value::CreateTensor<float>(memoryInfo,
                    pred.data(),
                    pred.size(),
                    outputShape.data(),
                    outputShape.size());
            const OnnxRunSlot slot;
            pSession->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, &output, 1);
            return true;
        } catch (const Ort::Exception& e) {
            qWarning() << "Beat This!: inference failed:" << e.what();
            return false;
        }
    };
    const std::vector<float> logits = beatLogits(m_resampler.audio(), runChunk);
    m_resampler.audio() = {};
    if (logits.empty()) {
        return false;
    }
    const std::vector<double> beats = pickBeats(logits);
    m_resultBeats.reserve(static_cast<int>(beats.size()));
    for (double seconds : beats) {
        m_resultBeats.push_back(mixxx::audio::FramePos(seconds * m_sampleRate.value()));
    }
    return true;
}

} // namespace mixxx
