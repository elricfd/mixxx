#pragma once

#include <memory>
#include <vector>

class FFTReal;
template<typename T>
class Window;

namespace mixxx {

/// Onset strength of the kick-drum band: half-wave rectified log-magnitude
/// spectral flux summed over 60-100 Hz. The broadband complex-domain
/// detection function is dominated by hi-hats and other off-beat
/// subdivisions in dance music, whose periodicity cannot tell the beat from
/// 1.5 or 0.5 beats. The bass band repeats at the beat, so its periodicity
/// tells the tempo tracker which level of the metre is the beat.
///
/// Kept free of Qt so it can be built into standalone analysis tools.
class BassOnsetFunction {
  public:
    /// windowSize must be even; ~2048 at 44.1 kHz gives ~21 Hz bins.
    BassOnsetFunction(double sampleRate, int windowSize);
    ~BassOnsetFunction();

    /// One mono frame of windowSize samples -> onset strength (>= 0).
    double processTimeDomain(const double* pSamples);

  private:
    const int m_windowSize;
    int m_firstBin;
    int m_endBin;
    std::unique_ptr<FFTReal> m_pFft;
    std::unique_ptr<Window<double>> m_pWindow;
    std::vector<double> m_frame;
    std::vector<double> m_magnitudes;
    std::vector<double> m_previousLogMagnitudes;
};

} // namespace mixxx
