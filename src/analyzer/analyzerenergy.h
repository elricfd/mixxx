#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "analyzer/analyzer.h"

class FFTReal;
struct SRC_STATE_tag;

namespace mixxx::energy {

constexpr int kSampleRate = 22050;
constexpr int kFrameSize = 2048;
constexpr int kHopSize = 1024;
using Features = std::array<double, 7>; // order of energymodel.h

/// Running feature extraction over mono audio at kSampleRate, matching
/// mik-tempo-research/energy_fit.py:features_of().
class FeatureAccumulator {
  public:
    FeatureAccumulator();
    ~FeatureAccumulator();
    void addSamples(const float* pMono, std::size_t count);
    /// std::nullopt if fewer than 10 non-silent frames were seen.
    std::optional<Features> features() const;

  private:
    void processFrame(const float* pFrame);

    std::unique_ptr<FFTReal> m_pFft;
    std::vector<float> m_buffer;
    std::size_t m_bufferStart;
    std::vector<double> m_window;
    std::vector<double> m_prevLogMag;
    std::vector<double> m_frameDb;
    double m_sumFlux;
    int m_fluxFrames;
    double m_sumCentroidHz;
    double m_sumLow;
    double m_sumHigh;
    int m_frames;
};

/// Energy level 1..10 from the trees in energymodel.h (fitted to Mixed In Key ratings).
int predictEnergy(const Features& features);

} // namespace mixxx::energy

/// Rates the track's energy 1..10 like Mixed In Key and stores it as text
/// (the EnergyLevel tag). Tracks that already have an energy, e.g. from MIK's
/// tags, are left alone.
class AnalyzerEnergy : public Analyzer {
  public:
    AnalyzerEnergy();
    ~AnalyzerEnergy() override;

    bool initialize(const AnalyzerTrack& track,
            mixxx::audio::SampleRate sampleRate,
            mixxx::audio::ChannelCount channelCount,
            SINT frameLength) override;
    bool processSamples(const CSAMPLE* pIn, SINT count) override;
    void storeResults(TrackPointer pTrack) override;
    void cleanup() override;

  private:
    bool resample(const float* pMono, long frames, bool endOfInput);

    SRC_STATE_tag* m_pResampler;
    double m_ratio;
    int m_channels;
    std::unique_ptr<mixxx::energy::FeatureAccumulator> m_pFeatures;
};
