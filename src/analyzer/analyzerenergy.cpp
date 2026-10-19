#include "analyzer/analyzerenergy.h"

#include <dsp/transforms/FFT.h>
#include <samplerate.h>

#include <QtDebug>
#include <algorithm>
#include <cmath>
#include <numbers>

#include "analyzer/analyzertrack.h"
#include "analyzer/plugins/energymodel.h"
#include "track/track.h"

namespace mixxx::energy {

namespace {
constexpr int kBins = kFrameSize / 2 + 1;
constexpr double kSilenceDb = -60.0;
constexpr int kMinFrames = 10;
} // namespace

FeatureAccumulator::FeatureAccumulator()
        : m_pFft(std::make_unique<FFTReal>(kFrameSize)),
          m_bufferStart(0),
          m_window(kFrameSize),
          m_sumFlux(0),
          m_fluxFrames(0),
          m_sumCentroidHz(0),
          m_sumLow(0),
          m_sumHigh(0),
          m_frames(0) {
    // numpy.hanning: symmetric Hann window
    for (int n = 0; n < kFrameSize; ++n) {
        m_window[n] = 0.5 - 0.5 * std::cos(2 * std::numbers::pi * n / (kFrameSize - 1));
    }
}

FeatureAccumulator::~FeatureAccumulator() = default;

void FeatureAccumulator::addSamples(const float* pMono, std::size_t count) {
    m_buffer.insert(m_buffer.end(), pMono, pMono + count);
    while (m_buffer.size() - m_bufferStart >= kFrameSize) {
        processFrame(m_buffer.data() + m_bufferStart);
        m_bufferStart += kHopSize;
    }
    if (m_bufferStart > (1 << 16)) {
        m_buffer.erase(m_buffer.begin(), m_buffer.begin() + m_bufferStart);
        m_bufferStart = 0;
    }
}

void FeatureAccumulator::processFrame(const float* pFrame) {
    std::array<double, kFrameSize> in;
    std::array<double, kFrameSize> re;
    std::array<double, kFrameSize> im;
    double meanSquare = 0;
    for (int n = 0; n < kFrameSize; ++n) {
        meanSquare += static_cast<double>(pFrame[n]) * pFrame[n];
        in[n] = pFrame[n] * m_window[n];
    }
    meanSquare /= kFrameSize;
    m_pFft->forward(in.data(), re.data(), im.data());

    std::vector<double> logMag(kBins);
    double total = 1e-12;
    double weighted = 0;
    double low = 0;
    double high = 0;
    for (int k = 0; k < kBins; ++k) {
        const double power = re[k] * re[k] + im[k] * im[k];
        const double hz = static_cast<double>(k) * kSampleRate / kFrameSize;
        logMag[k] = std::log1p(100 * std::sqrt(power));
        total += power;
        weighted += power * hz;
        low += hz < 150 ? power : 0;
        high += hz > 4000 ? power : 0;
    }
    const double db = 10 * std::log10(meanSquare + 1e-12);
    const bool keep = db > kSilenceDb;
    if (!m_prevLogMag.empty() && keep) {
        double flux = 0;
        for (int k = 0; k < kBins; ++k) {
            flux += std::max(logMag[k] - m_prevLogMag[k], 0.0);
        }
        m_sumFlux += flux;
        ++m_fluxFrames;
    }
    if (keep) {
        m_frameDb.push_back(db);
        m_sumCentroidHz += weighted / total;
        m_sumLow += low / total;
        m_sumHigh += high / total;
    }
    m_prevLogMag = std::move(logMag);
    ++m_frames;
}

std::optional<Features> FeatureAccumulator::features() const {
    const auto kept = static_cast<int>(m_frameDb.size());
    if (m_frames < kMinFrames || kept < kMinFrames) {
        return std::nullopt;
    }
    double mean = 0;
    for (double db : m_frameDb) {
        mean += db;
    }
    mean /= kept;
    double var = 0;
    for (double db : m_frameDb) {
        var += (db - mean) * (db - mean);
    }
    // numpy.percentile(..., 90) with linear interpolation
    std::vector<double> sorted = m_frameDb;
    std::sort(sorted.begin(), sorted.end());
    const double pos = 0.9 * (kept - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const double p90 = sorted[lo] + (pos - lo) * (sorted[std::min<std::size_t>(lo + 1, kept - 1)] - sorted[lo]);
    return Features{mean,
            p90,
            std::sqrt(var / kept),
            m_fluxFrames > 0 ? m_sumFlux / m_fluxFrames : 0.0,
            m_sumCentroidHz / kept / 1000,
            m_sumLow / kept,
            m_sumHigh / kept};
}

int predictEnergy(const Features& features) {
    double sum = energymodel::kBaseline;
    for (const int root : energymodel::kRoots) {
        int i = root;
        while (energymodel::kNodes[i].feature >= 0) {
            const auto& node = energymodel::kNodes[i];
            i = features[node.feature] <= node.threshold ? node.left : node.right;
        }
        sum += energymodel::kNodes[i].value;
    }
    return static_cast<int>(std::clamp(std::nearbyint(sum), 1.0, 10.0));
}

} // namespace mixxx::energy

AnalyzerEnergy::AnalyzerEnergy()
        : m_pResampler(nullptr),
          m_ratio(1),
          m_channels(0) {
}

AnalyzerEnergy::~AnalyzerEnergy() {
    cleanup();
}

bool AnalyzerEnergy::initialize(const AnalyzerTrack& track,
        mixxx::audio::SampleRate sampleRate,
        mixxx::audio::ChannelCount channelCount,
        SINT frameLength) {
    if (frameLength <= 0 || !track.getTrack()->getEnergy().isEmpty()) {
        return false;
    }
    int error = 0;
    m_pResampler = src_new(SRC_SINC_MEDIUM_QUALITY, 1, &error);
    if (!m_pResampler) {
        qWarning() << "AnalyzerEnergy: resampler failed:" << src_strerror(error);
        return false;
    }
    m_ratio = static_cast<double>(mixxx::energy::kSampleRate) / sampleRate.value();
    m_channels = channelCount;
    m_pFeatures = std::make_unique<mixxx::energy::FeatureAccumulator>();
    return true;
}

bool AnalyzerEnergy::resample(const float* pMono, long frames, bool endOfInput) {
    float out[8192];
    SRC_DATA data{};
    data.data_in = pMono;
    data.input_frames = frames;
    data.src_ratio = m_ratio;
    data.end_of_input = endOfInput ? 1 : 0;
    do {
        data.data_out = out;
        data.output_frames = static_cast<long>(std::size(out));
        const int error = src_process(m_pResampler, &data);
        if (error) {
            qWarning() << "AnalyzerEnergy: resampling failed:" << src_strerror(error);
            return false;
        }
        m_pFeatures->addSamples(out, data.output_frames_gen);
        data.data_in += data.input_frames_used;
        data.input_frames -= data.input_frames_used;
    } while (data.input_frames > 0 || (endOfInput && data.output_frames_gen > 0));
    return true;
}

bool AnalyzerEnergy::processSamples(const CSAMPLE* pIn, SINT count) {
    VERIFY_OR_DEBUG_ASSERT(m_pResampler && m_channels > 0) {
        return false;
    }
    // Stereo -> (L + R) / 2; stem files carry stereo pairs that sum to the mix.
    const double scale = m_channels == 1 ? 1.0 : 0.5;
    const SINT frames = count / m_channels;
    std::vector<float> mono(frames);
    for (SINT f = 0; f < frames; ++f) {
        double sum = 0;
        for (int c = 0; c < m_channels; ++c) {
            sum += pIn[f * m_channels + c];
        }
        mono[f] = static_cast<float>(sum * scale);
    }
    return resample(mono.data(), frames, false);
}

void AnalyzerEnergy::storeResults(TrackPointer pTrack) {
    VERIFY_OR_DEBUG_ASSERT(m_pResampler) {
        return;
    }
    resample(nullptr, 0, true);
    const auto features = m_pFeatures->features();
    if (!features) {
        return;
    }
    pTrack->setEnergy(QString::number(mixxx::energy::predictEnergy(*features)));
}

void AnalyzerEnergy::cleanup() {
    if (m_pResampler) {
        src_delete(m_pResampler);
        m_pResampler = nullptr;
    }
    m_pFeatures.reset();
}
