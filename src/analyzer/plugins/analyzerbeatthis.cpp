#include "analyzer/plugins/analyzerbeatthis.h"

#include <onnxruntime_cxx_api.h>
#include <samplerate.h>

#include <QDir>
#include <QFileInfo>
#include <QMutex>
#include <QtDebug>
#include <array>
#include <map>
#include <memory>

#include "analyzer/constants.h"

namespace mixxx {
namespace {

Ort::Env& ortEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mixxx-beatthis");
    return env;
}

/// One session per model file, shared by all analyzer threads
/// (Ort::Session::Run is thread-safe). Loading takes ~1 s, so keep it.
std::shared_ptr<Ort::Session> session(const QString& modelPath) {
    static QMutex mutex;
    static std::map<QString, std::shared_ptr<Ort::Session>> sessions;
    const auto locker = QMutexLocker(&mutex);
    auto it = sessions.find(modelPath);
    if (it != sessions.end()) {
        return it->second;
    }
    std::shared_ptr<Ort::Session> pSession;
    try {
        Ort::SessionOptions options;
        // Several tracks are analyzed in parallel; keep each one modest.
        options.SetIntraOpNumThreads(2);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        const std::wstring path = QDir::toNativeSeparators(modelPath).toStdWString();
#else
        const std::string path = modelPath.toStdString();
#endif
        pSession = std::make_shared<Ort::Session>(ortEnv(), path.c_str(), options);
    } catch (const Ort::Exception& e) {
        qWarning() << "Beat This!: failed to load model" << modelPath << e.what();
    }
    sessions[modelPath] = pSession; // also cache failures
    return pSession;
}

} // namespace

// static
QString AnalyzerBeatThis::modelPath(const QString& resourcePath) {
    return QDir(resourcePath).filePath(QStringLiteral("models/beatthis_final0.onnx"));
}

AnalyzerBeatThis::AnalyzerBeatThis(const QString& modelPath)
        : m_modelPath(modelPath),
          m_pResampler(nullptr) {
}

AnalyzerBeatThis::~AnalyzerBeatThis() {
    if (m_pResampler) {
        src_delete(m_pResampler);
    }
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
    m_audio.clear();
    m_sampleRate = sampleRate;
    if (!QFileInfo::exists(m_modelPath)) {
        qWarning() << "Beat This!: model file not found:" << m_modelPath;
        return false;
    }
    if (m_pResampler) {
        src_delete(m_pResampler);
    }
    int error = 0;
    m_pResampler = src_new(SRC_SINC_MEDIUM_QUALITY, 1, &error);
    if (!m_pResampler) {
        qWarning() << "Beat This!: resampler failed:" << src_strerror(error);
        return false;
    }
    // ~10 minutes at 22050 Hz; grows if needed.
    m_audio.reserve(static_cast<std::size_t>(kModelSampleRate) * 600);
    return true;
}

bool AnalyzerBeatThis::resample(const float* pMono, long frames, bool endOfInput) {
    const double ratio = static_cast<double>(kModelSampleRate) / m_sampleRate.value();
    float out[8192];
    SRC_DATA data{};
    data.data_in = pMono;
    data.input_frames = frames;
    data.src_ratio = ratio;
    data.end_of_input = endOfInput ? 1 : 0;
    do {
        data.data_out = out;
        data.output_frames = static_cast<long>(std::size(out));
        const int error = src_process(m_pResampler, &data);
        if (error) {
            qWarning() << "Beat This!: resampling failed:" << src_strerror(error);
            return false;
        }
        m_audio.insert(m_audio.end(), out, out + data.output_frames_gen);
        data.data_in += data.input_frames_used;
        data.input_frames -= data.input_frames_used;
    } while (data.input_frames > 0 || (endOfInput && data.output_frames_gen > 0));
    return true;
}

bool AnalyzerBeatThis::processSamples(const CSAMPLE* pIn, SINT iLen) {
    DEBUG_ASSERT(iLen % kAnalysisChannels == 0);
    if (!m_pResampler) {
        return false;
    }
    const SINT frames = iLen / kAnalysisChannels;
    m_mono.resize(frames);
    for (SINT i = 0; i < frames; ++i) {
        m_mono[i] = (pIn[2 * i] + pIn[2 * i + 1]) * 0.5f;
    }
    return resample(m_mono.data(), static_cast<long>(frames), false);
}

bool AnalyzerBeatThis::finalize() {
    if (!m_pResampler || !resample(nullptr, 0, true)) {
        return false;
    }
    src_delete(m_pResampler);
    m_pResampler = nullptr;

    const std::shared_ptr<Ort::Session> pSession = session(m_modelPath);
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
            pSession->Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, &output, 1);
            return true;
        } catch (const Ort::Exception& e) {
            qWarning() << "Beat This!: inference failed:" << e.what();
            return false;
        }
    };
    const std::vector<float> logits = beatLogits(m_audio, runChunk);
    m_audio = {};
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
