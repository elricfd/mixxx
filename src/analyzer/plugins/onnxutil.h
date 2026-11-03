#pragma once

#include <QString>
#include <memory>
#include <vector>

#include "audio/types.h"
#include "util/types.h"

struct SRC_STATE_tag;

namespace Ort {
struct Session;
} // namespace Ort

namespace mixxx {

/// One ONNX Runtime session per model file, shared by all analyzer threads
/// (Ort::Session::Run is thread-safe). Returns null if the model fails to load.
std::shared_ptr<Ort::Session> onnxSession(const QString& modelPath);

/// Hold one while calling Ort::Session::Run. Mixxx analyzes many tracks at once
/// (one thread per core) and a Beat This! run peaks near 800 MB, so at most
/// kMaxConcurrentRuns models run at the same time.
class OnnxRunSlot {
  public:
    // ponytail: fixed at 4 (x 2 intra-op threads); make it a preference if machines differ a lot
    static constexpr int kMaxConcurrentRuns = 4;
    OnnxRunSlot();
    ~OnnxRunSlot();
    OnnxRunSlot(const OnnxRunSlot&) = delete;
    OnnxRunSlot& operator=(const OnnxRunSlot&) = delete;
};

/// Collects a whole track as mono float at a fixed model sample rate:
/// interleaved stereo in, averaged, resampled with libsamplerate.
class MonoResampler {
  public:
    explicit MonoResampler(int targetRate)
            : m_targetRate(targetRate) {
    }
    ~MonoResampler();

    bool initialize(mixxx::audio::SampleRate sampleRate);
    bool process(const CSAMPLE* pStereo, SINT len);
    /// Flushes the resampler; audio() is complete afterwards.
    bool finish();

    std::vector<float>& audio() {
        return m_audio;
    }

  private:
    bool resample(const float* pMono, long frames, bool endOfInput);

    const int m_targetRate;
    mixxx::audio::SampleRate m_sampleRate;
    SRC_STATE_tag* m_pState = nullptr;
    std::vector<float> m_mono;
    std::vector<float> m_audio;
};

} // namespace mixxx
