#include "analyzer/plugins/bassonsetfunction.h"

#include <base/Window.h>
#include <dsp/transforms/FFT.h>

#include <algorithm>
#include <cmath>

namespace mixxx {
namespace {

constexpr double kLowestHz = 60.0;
constexpr double kHighestHz = 100.0;
// Log compression of the magnitudes, so the flux follows the relative
// change of a kick rather than the loudness of the track.
constexpr double kCompression = 100.0;

} // namespace

BassOnsetFunction::BassOnsetFunction(double sampleRate, int windowSize)
        : m_windowSize(windowSize),
          m_firstBin(static_cast<int>(std::ceil(kLowestHz * windowSize / sampleRate))),
          m_endBin(static_cast<int>(std::ceil(kHighestHz * windowSize / sampleRate))),
          m_pFft(std::make_unique<FFTReal>(windowSize)),
          m_pWindow(std::make_unique<Window<double>>(HanningWindow, windowSize)),
          m_frame(windowSize),
          m_magnitudes(windowSize),
          m_previousLogMagnitudes(std::max(m_endBin - m_firstBin, 0)) {
}

BassOnsetFunction::~BassOnsetFunction() = default;

double BassOnsetFunction::processTimeDomain(const double* pSamples) {
    m_pWindow->cut(pSamples, m_frame.data());
    m_pFft->forwardMagnitude(m_frame.data(), m_magnitudes.data());
    double flux = 0.0;
    for (int bin = m_firstBin; bin < m_endBin; ++bin) {
        const double logMagnitude = std::log1p(kCompression * m_magnitudes[bin]);
        double& previous = m_previousLogMagnitudes[bin - m_firstBin];
        flux += std::max(logMagnitude - previous, 0.0);
        previous = logMagnitude;
    }
    return flux;
}

} // namespace mixxx
