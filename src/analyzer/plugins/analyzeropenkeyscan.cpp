#include "analyzer/plugins/analyzeropenkeyscan.h"

#include <onnxruntime_cxx_api.h>

#include <QDir>
#include <QFileInfo>
#include <QtDebug>
#include <algorithm>
#include <array>

namespace mixxx {

using mixxx::track::io::key::ChromaticKey;

// static
QString AnalyzerOpenKeyScan::modelPath(const QString& resourcePath) {
    return QDir(resourcePath).filePath(QStringLiteral("models/openkeyscan3.onnx"));
}

// static
ChromaticKey AnalyzerOpenKeyScan::keyFromIndex(int index) {
    if (index < 0 || index >= 24) {
        return ChromaticKey::INVALID;
    }
    // Camelot nB: major tonic = (n - 8) * 7 semitones above C; nA is its relative minor.
    const int majorTonic = ((index % 12 + 1 - 8) * 7 % 12 + 12) % 12;
    if (index >= 12) {
        return static_cast<ChromaticKey>(ChromaticKey::C_MAJOR + majorTonic);
    }
    return static_cast<ChromaticKey>(ChromaticKey::C_MINOR + (majorTonic + 9) % 12);
}

AnalyzerOpenKeyScan::AnalyzerOpenKeyScan(const QString& modelPath)
        : m_modelPath(modelPath),
          m_resampler(kModelSampleRate) {
}

bool AnalyzerOpenKeyScan::initialize(mixxx::audio::SampleRate sampleRate) {
    m_resultKeys.clear();
    if (!QFileInfo::exists(m_modelPath)) {
        qWarning() << "OpenKeyScan: model file not found:" << m_modelPath;
        return false;
    }
    return m_resampler.initialize(sampleRate);
}

bool AnalyzerOpenKeyScan::processSamples(const CSAMPLE* pIn, SINT iLen) {
    return m_resampler.process(pIn, iLen);
}

bool AnalyzerOpenKeyScan::finalize() {
    if (!m_resampler.finish()) {
        return false;
    }
    std::vector<float>& audio = m_resampler.audio();
    // The CNN pools 8x over 5 frames/s; shorter clips have no meaningful key anyway.
    if (audio.size() < static_cast<std::size_t>(kModelSampleRate) * 5) {
        return false;
    }
    const std::shared_ptr<Ort::Session> pSession = onnxSession(m_modelPath);
    if (!pSession) {
        return false;
    }
    std::array<float, 24> logits{};
    try {
        const Ort::MemoryInfo memoryInfo =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 2> inputShape{1, static_cast<int64_t>(audio.size())};
        const std::array<int64_t, 2> outputShape{1, 24};
        const char* inputNames[] = {"audio"};
        const char* outputNames[] = {"key"};
        Ort::Value input = Ort::Value::CreateTensor<float>(memoryInfo,
                audio.data(),
                audio.size(),
                inputShape.data(),
                inputShape.size());
        Ort::Value output = Ort::Value::CreateTensor<float>(memoryInfo,
                logits.data(),
                logits.size(),
                outputShape.data(),
                outputShape.size());
        const OnnxRunSlot slot;
        pSession->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, &output, 1);
    } catch (const Ort::Exception& e) {
        qWarning() << "OpenKeyScan: inference failed:" << e.what();
        return false;
    }
    audio = {};
    const int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    m_resultKeys.push_back(qMakePair(keyFromIndex(best), 0.0));
    return true;
}

} // namespace mixxx
