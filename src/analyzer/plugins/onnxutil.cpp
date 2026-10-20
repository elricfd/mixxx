#include "analyzer/plugins/onnxutil.h"

#include <onnxruntime_cxx_api.h>
#include <samplerate.h>

#include <QDir>
#include <QMutex>
#include <QSemaphore>
#include <QtDebug>
#include <map>

#include "analyzer/constants.h"

namespace mixxx {

std::shared_ptr<Ort::Session> onnxSession(const QString& modelPath) {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mixxx-analyzer");
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
        // The arena keeps the peak of all concurrent runs forever (15 GB after a
        // 26-track batch); without it memory goes back after every run.
        options.DisableCpuMemArena();
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        const std::wstring path = QDir::toNativeSeparators(modelPath).toStdWString();
#else
        const std::string path = modelPath.toStdString();
#endif
        pSession = std::make_shared<Ort::Session>(env, path.c_str(), options);
    } catch (const Ort::Exception& e) {
        qWarning() << "ONNX Runtime: failed to load model" << modelPath << e.what();
    }
    sessions[modelPath] = pSession; // also cache failures
    return pSession;
}

namespace {
QSemaphore s_runSlots(OnnxRunSlot::kMaxConcurrentRuns);
} // namespace

OnnxRunSlot::OnnxRunSlot() {
    s_runSlots.acquire();
}

OnnxRunSlot::~OnnxRunSlot() {
    s_runSlots.release();
}

MonoResampler::~MonoResampler() {
    if (m_pState) {
        src_delete(m_pState);
    }
}

bool MonoResampler::initialize(mixxx::audio::SampleRate sampleRate) {
    m_sampleRate = sampleRate;
    m_audio.clear();
    if (m_pState) {
        src_delete(m_pState);
    }
    int error = 0;
    m_pState = src_new(SRC_SINC_MEDIUM_QUALITY, 1, &error);
    if (!m_pState) {
        qWarning() << "Resampler failed:" << src_strerror(error);
        return false;
    }
    // ~10 minutes; grows if needed.
    m_audio.reserve(static_cast<std::size_t>(m_targetRate) * 600);
    return true;
}

bool MonoResampler::resample(const float* pMono, long frames, bool endOfInput) {
    float out[8192];
    SRC_DATA data{};
    data.data_in = pMono;
    data.input_frames = frames;
    data.src_ratio = static_cast<double>(m_targetRate) / m_sampleRate.value();
    data.end_of_input = endOfInput ? 1 : 0;
    do {
        data.data_out = out;
        data.output_frames = static_cast<long>(std::size(out));
        const int error = src_process(m_pState, &data);
        if (error) {
            qWarning() << "Resampling failed:" << src_strerror(error);
            return false;
        }
        m_audio.insert(m_audio.end(), out, out + data.output_frames_gen);
        data.data_in += data.input_frames_used;
        data.input_frames -= data.input_frames_used;
    } while (data.input_frames > 0 || (endOfInput && data.output_frames_gen > 0));
    return true;
}

bool MonoResampler::process(const CSAMPLE* pStereo, SINT len) {
    DEBUG_ASSERT(len % kAnalysisChannels == 0);
    if (!m_pState) {
        return false;
    }
    const SINT frames = len / kAnalysisChannels;
    m_mono.resize(frames);
    for (SINT i = 0; i < frames; ++i) {
        m_mono[i] = (pStereo[2 * i] + pStereo[2 * i + 1]) * 0.5f;
    }
    return resample(m_mono.data(), static_cast<long>(frames), false);
}

bool MonoResampler::finish() {
    if (!m_pState || !resample(nullptr, 0, true)) {
        return false;
    }
    src_delete(m_pState);
    m_pState = nullptr;
    return true;
}

} // namespace mixxx
