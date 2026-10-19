/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    QM DSP Library

    Centre for Digital Music, Queen Mary, University of London.
    This file copyright 2008-2009 Matthew Davies and QMUL.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#include "TempoTrackV2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "maths/MathUtilities.h"

using std::vector;

#define   EPS 0.0000008 // just some arbitrary small number

TempoTrackV2::TempoTrackV2(float rate, int increment) :
    m_rate(rate), m_increment(increment) {
}

TempoTrackV2::~TempoTrackV2() { }

void
TempoTrackV2::filter_df(d_vec_t &df)
{
    int df_len = int(df.size());
    
    d_vec_t a(3);
    d_vec_t b(3);
    d_vec_t lp_df(df_len);

    //equivalent in matlab to [b,a] = butter(2,0.4);
    a[0] = 1.0000;
    a[1] = -0.3695;
    a[2] = 0.1958;
    b[0] = 0.2066;
    b[1] = 0.4131;
    b[2] = 0.2066;

    double inp1 = 0.;
    double inp2 = 0.;
    double out1 = 0.;
    double out2 = 0.;


    // forwards filtering
    for (int i = 0; i < df_len; i++) {
        lp_df[i] =  b[0]*df[i] + b[1]*inp1 + b[2]*inp2 - a[1]*out1 - a[2]*out2;
        inp2 = inp1;
        inp1 = df[i];
        out2 = out1;
        out1 = lp_df[i];
    }

    // copy forwards filtering to df...
    // but, time-reversed, ready for backwards filtering
    for (int i = 0; i < df_len; i++) {
        df[i] = lp_df[df_len - i - 1];
    }

    for (int i = 0; i < df_len; i++) {
        lp_df[i] = 0.;
    }

    inp1 = 0.; inp2 = 0.;
    out1 = 0.; out2 = 0.;

    // backwards filetering on time-reversed df
    for (int i = 0; i < df_len; i++) {
        lp_df[i] =  b[0]*df[i] + b[1]*inp1 + b[2]*inp2 - a[1]*out1 - a[2]*out2;
        inp2 = inp1;
        inp1 = df[i];
        out2 = out1;
        out1 = lp_df[i];
    }

    // write the re-reversed (i.e. forward) version back to df
    for (int i = 0; i < df_len; i++) {
        df[i] = lp_df[df_len - i - 1];
    }
}


// MEPD 28/11/12
// This function now allows for a user to specify an inputtempo (in BPM)
// and a flag "constraintempo" which replaces the general rayleigh weighting for periodicities
// with a gaussian which is centered around the input tempo
// Note, if inputtempo = 120 and constraintempo = false, then functionality is
// as it was before
void
TempoTrackV2::calculateBeatPeriod(const vector<double> &df,
                                  vector<int> &beat_period,
                                  double inputtempo, bool constraintempo)
{
    // to follow matlab.. split into 512 sample frames with a 128 hop size
    // calculate the acf,
    // then the rcf.. and then stick the rcfs as columns of a matrix
    // then call viterbi decoding with weight vector and transition matrix
    // and get best path

    d_vec_t wv;
    get_wv(inputtempo, constraintempo, wv);

    d_mat_t rcfmat;
    get_rcfmat(df, wv, rcfmat);

    // now call viterbi decoding function
    viterbi_decode(rcfmat,wv,beat_period);
}


void
TempoTrackV2::get_wv(double inputtempo, bool constraintempo, d_vec_t &wv)
{
    const int wv_len = 128;

    // MEPD 28/11/12
    // the default value of inputtempo in the beat tracking plugin is 120
    // so if the user specifies a different inputtempo, the rayparam will be updated
    // accordingly.
    // note: 60*44100/512 is a magic number
    // this might (will?) break if a user specifies a different frame rate for the onset detection function
    const double rayparam = (60 * 44100 / 512.0) / inputtempo;

    // make rayleigh weighting curve
    wv.assign(wv_len, 0.);

    // check whether or not to use rayleigh weighting (if constraintempo is false)
    // or use gaussian weighting it (constraintempo is true)
    if (constraintempo) {
        for (int i = 0; i < wv_len; i++) {
            // MEPD 28/11/12
            // do a gaussian weighting instead of rayleigh
            wv[i] = exp( (-1.*pow((double(i)-rayparam),2.)) / (2.*pow(rayparam/4.,2.)) );
        }
    } else {
        for (int i = 0; i < wv_len; i++) {
            // MEPD 28/11/12
            // standard rayleigh weighting over periodicities
            wv[i] = (double(i) / pow(rayparam,2.)) * exp((-1.*pow(-double(i),2.)) / (2.*pow(rayparam,2.)));
        }
    }
}


void
TempoTrackV2::get_rcfmat(const d_vec_t &df, const d_vec_t &wv, d_mat_t &rcfmat)
{
    const int wv_len = int(wv.size());

    // Beat tracking window length (roughly 6 seconds) and hop size (1.5 seconds)
    int winlen = 512;
    int hopsize = 128;
    
    int df_len = int(df.size());
    
    // Matrix to store output of comb filter bank
    rcfmat.clear();
    rcfmat.reserve(df_len / hopsize + 1);
    d_vec_t dfframe(winlen);
    d_vec_t rcf(wv_len);

    // Loop over the onset detection function half a window padding on both ends
    for (int i = -winlen / 2; i < df_len - winlen / 2; i += hopsize) {
        int k = 0;
        int l = winlen;

        if (i < 0) {
            k = -i;
            std::fill(dfframe.begin(), dfframe.begin() + k, 0.0);
        }

        if (i + l > df_len) {
            l = df_len - i;
            std::fill(dfframe.begin() + l, dfframe.end(), 0.0);
        }
        
        std::copy(df.begin() + i + k, df.begin() + i + l,
                      dfframe.begin() + k);

        // Apply the resonator comb filter (RCF) bank to the window
        // The result is a vector of filter responses for different periods.
        // Mixxx: get_rcf() accumulates into rcf, so start each window from
        // zero (as upstream qm-dsp does by allocating it per window) instead
        // of on top of the previous window's normalised response.
        std::fill(rcf.begin(), rcf.end(), 0.0);
        get_rcf(dfframe, wv, rcf);

        // Append the result to rcfmat as a new column 
        rcfmat.push_back(d_vec_t());
        for (int j = 0; j < wv_len; j++) {
            rcfmat.back().push_back(rcf[j]);
        }
    }
}

namespace {

// Median of the first n entries (the rows that viterbi_decode() filled).
double median_period(std::vector<int> periods, std::size_t n)
{
    if (n == 0 || n > periods.size()) {
        return 0.;
    }
    periods.resize(n);
    std::sort(periods.begin(), periods.end());
    return n % 2 ? periods[n / 2] : 0.5 * (periods[n / 2 - 1] + periods[n / 2]);
}

} // namespace

// Mixxx: tempo from two onset detection functions of the same frame rate,
// e.g. the broadband df and a bass-band df.
//
// The broadband df of dance music is often dominated by off-beat
// subdivisions such as hi-hats, whose periodicity cannot tell a beat period
// from one of 1.5 beats (or 0.75 beats from one beat), so the tracker locks
// to 2/3 or 4/3 of the tempo. Multiplying in the periodicity distribution of
// the bass-band df, which repeats at the beat, resolves that duple/triple
// ambiguity. Bass lines are not a reliable guide to the octave (half-time
// and eighth-note bass is common), so the combined path is only used when it
// moves the beat period by such a triple ratio; otherwise the broadband
// result is kept unchanged.
void
TempoTrackV2::calculateBeatPeriod(const vector<double> &df,
                                  const vector<double> &df2,
                                  vector<int> &beat_period)
{
    d_vec_t wv;
    get_wv(120.0, false, wv);
    const int wv_len = int(wv.size());

    d_mat_t rcfmat;
    get_rcfmat(df, wv, rcfmat);
    d_mat_t rcfmat2;
    get_rcfmat(df2, wv, rcfmat2);

    i_vec_t broadband(beat_period);
    viterbi_decode(rcfmat, wv, broadband);

    const std::size_t T = std::min(rcfmat.size(), rcfmat2.size());
    rcfmat.resize(T);
    for (std::size_t t = 0; t < T; t++) {
        double sum = 0.;
        for (int j = 0; j < wv_len; j++) {
            rcfmat[t][j] *= rcfmat2[t][j];
            sum += rcfmat[t][j];
        }
        for (int j = 0; j < wv_len; j++) {
            rcfmat[t][j] /= (sum + EPS);
        }
    }
    i_vec_t combined(beat_period);
    viterbi_decode(rcfmat, wv, combined);

    const double broadbandPeriod = median_period(broadband, T);
    const double combinedPeriod = median_period(combined, T);
    bool triple = false;
    if (broadbandPeriod > 0. && combinedPeriod > 0.) {
        const double ratio = combinedPeriod / broadbandPeriod;
        const double tolerance = 0.05;
        triple = std::fabs(ratio / (2. / 3.) - 1.) <= tolerance ||
                std::fabs(ratio / (4. / 3.) - 1.) <= tolerance;
    }
    beat_period = triple ? combined : broadband;
}

void
TempoTrackV2::get_rcf(const d_vec_t &dfframe_in, const d_vec_t &wv, d_vec_t &rcf)
{
    // calculate autocorrelation function
    // then rcf
    // just hard code for now... don't really need separate functions to do this

    // make acf

    d_vec_t dfframe(dfframe_in);

    MathUtilities::adaptiveThreshold(dfframe);

    int dfframe_len = int(dfframe.size());
    int rcf_len = int(rcf.size());
    
    d_vec_t acf(dfframe_len);

    for (int lag = 0; lag < dfframe_len; lag++) {
        double sum = 0.;
        double tmp = 0.;

        for (int n = 0; n < (dfframe_len - lag); n++) {
            tmp = dfframe[n] * dfframe[n + lag];
            sum += tmp;
        }
        acf[lag] = double(sum/ (dfframe_len - lag));
    }

    // now apply comb filtering
    int numelem = 4;

    for (int i = 2; i < rcf_len; i++) { // max beat period
        for (int a = 1; a <= numelem; a++) { // number of comb elements
            for (int b = 1-a; b <= a-1; b++) { // general state using normalisation of comb elements
                rcf[i-1] += ( acf[(a*i+b)-1]*wv[i-1] ) / (2.*a-1.);     // calculate value for comb filter row
            }
        }
    }

    // apply adaptive threshold to rcf
    MathUtilities::adaptiveThreshold(rcf);

    double rcfsum =0.;
    for (int i = 0; i < rcf_len; i++) {
        rcf[i] += EPS ;
        rcfsum += rcf[i];
    }

    // normalise rcf to sum to unity
    for (int i = 0; i < rcf_len; i++) {
        rcf[i] /= (rcfsum + EPS);
    }
}

void
TempoTrackV2::viterbi_decode(const d_mat_t &rcfmat, const d_vec_t &wv, i_vec_t &beat_period)
{
    // following Kevin Murphy's Viterbi decoding to get best path of
    // beat periods through rfcmat

    if (rcfmat.size() < 2) return; // can't do anything at all meaningful

    const std::size_t T = rcfmat.size();
    const std::size_t Q = rcfmat[0].size();
 
    auto tmat = d_mat_t(Q, d_vec_t(Q));
    // variance of Gaussians in transition matrix
    // formed of Gaussians on diagonal - implies slow tempo change
    double sigma = 8.;
    // don't want really short beat periods, or really long ones
    for (std::size_t i = 20; i  < Q - 20; i++) {
        for (int j = 20; j < Q - 20; j++) {
            double mu = double(i);
            tmat[i][j] = exp( (-1.*pow((j-mu),2.)) / (2.*pow(sigma,2.)) );
        }
    }

    // parameters for Viterbi decoding... this part is taken from
    // Murphy's matlab

    auto delta = d_mat_t(T, d_vec_t(Q));
    auto psi = i_mat_t(T, i_vec_t(Q));

    // initialize first column of delta
    for (std::size_t j = 0; j < Q; j++) {
        delta[0][j] = wv[j] * rcfmat[0][j];
    }

    double deltasum = 0.;
    for (std::size_t i = 0; i < Q; i++) {
        deltasum += delta[0][i];
    }
    for (std::size_t i = 0; i < Q; i++) {
        delta[0][i] /= (deltasum + EPS);
    }

    for (std::size_t t=1; t < T; t++)
    {
        d_vec_t tmp_vec(Q);

        for (std::size_t j = 0; j < Q; j++) {
            for (std::size_t i = 0; i < Q; i++) {
                tmp_vec[i] = delta[t-1][i] * tmat[j][i];
            }

            delta[t][j] = get_max_val(tmp_vec);

            psi[t][j] = get_max_ind(tmp_vec);

            delta[t][j] *= rcfmat[t][j];
        }

        // normalise current delta column
        double deltasum = 0.;
        for (std::size_t i = 0; i < Q; i++) {
            deltasum += delta[t][i];
        }
        for (std::size_t i = 0; i < Q; i++) {
            delta[t][i] /= (deltasum + EPS);
        }
    }

    i_vec_t &bestpath = beat_period;
    // find starting point - best beat period for "last" frame
    bestpath[T-1] = get_max_ind(delta[T-1]);

    // backtrace through index of maximum values in psi
    for (int t=T-2; t>0 ;t--) {
        bestpath[t] = psi[t+1][bestpath[t+1]];
    }

    // weird but necessary hack -- couldn't get above loop to terminate at t >= 0
    bestpath[0] = psi[1][bestpath[1]];
}

double
TempoTrackV2::get_max_val(const d_vec_t &df)
{
    double maxval = 0.;
    int df_len = int(df.size());
    
    for (int i = 0; i < df_len; i++) {
        if (maxval < df[i]) {
            maxval = df[i];
        }
    }

    return maxval;
}

int
TempoTrackV2::get_max_ind(const d_vec_t &df)
{
    double maxval = 0.;
    int ind = 0;
    int df_len = int(df.size());
    
    for (int i = 0; i < df_len; i++) {
        if (maxval < df[i]) {
            maxval = df[i];
            ind = i;
        }
    }

    return ind;
}

void
TempoTrackV2::normalise_vec(d_vec_t &df)
{
    double sum = 0.;
    int df_len = int(df.size());
    
    for (int i = 0; i < df_len; i++) {
        sum += df[i];
    }

    for (int i = 0; i < df_len; i++) {
        df[i]/= (sum + EPS);
    }
}

// MEPD 28/11/12
// this function has been updated to allow the "alpha" and "tightness" parameters
// of the dynamic program to be set by the user
// the default value of alpha = 0.9 and tightness = 4
void
TempoTrackV2::calculateBeats(const vector<double> &df,
                             const vector<int> &beat_period,
                             vector<double> &beats, double alpha, double tightness)
{
    if (df.empty() || beat_period.empty()) return;

    int df_len = int(df.size());

    d_vec_t cumscore(df_len); // store cumulative score
    i_vec_t backlink(df_len); // backlink (stores best beat locations at each time instant)
    d_vec_t localscore(df_len); // localscore, for now this is the same as the detection function

    for (int i = 0; i < df_len; i++) {
        localscore[i] = df[i];
        backlink[i] = -1;
    }

    //double tightness = 4.;
    //double alpha = 0.9;
    // MEPD 28/11/12
    // debug statements that can be removed.
//    std::cerr << "alpha" << alpha << std::endl;
//    std::cerr << "tightness" << tightness << std::endl;

    int old_period = 0;
    int txwt_len = 0;
    d_vec_t txwt;

    // main loop
    for (int i = 0; i < df_len; i++) {
        // df contains the magnitude of the onsets
        //
        // beat_period is the viterbi path following the most likely bpm in a
        // 128 steps window of ~1,5 s. This is optimized to follow gentle tempo
        // changes that may happen in live recordings.
        //
        // This method is tricked by beat less bridges, changes in instrumentation
        // or grove or intended abrupt tempo changes. It may follow a previous
        // dominant even if we are in a new tempo window with lower onsets.

        int period = beat_period[i/128];
        int prange_min = period * -2;
        if (period != old_period)  {
            old_period = period;
            int prange_max = period / -2;

            // transition range
            txwt_len = prange_max - prange_min + 1;

	        txwt.clear();
	        txwt.reserve(txwt_len);
	        for (int j = 0; j < txwt_len; j++) {
	            double mu = double(period);
                txwt.push_back(exp( -0.5*pow(tightness * log((round(2*mu)-j)/mu),2)));
	        }
	    }
	    
        // loop over the region in cumscore of one period in the past
        // to check for resonance using the txwt pattern.
        double vv = 0; // the maximum value, high if there is likely beat
        int xx = 0; // index of the maximum
        for (int j = 0; j < txwt_len; j++) {
            int cscore_ind = i + prange_min + j;
            if (cscore_ind >= 0) {
                double scorecands = txwt[j] * cumscore[cscore_ind];
                if (scorecands > vv) {
                    vv = scorecands;
                    xx = cscore_ind;
                }
            }
        }

        cumscore[i] = alpha * vv + (1. - alpha) * localscore[i];
        backlink[i] = xx;

//        std::cerr << "backlink[" << i << "] <= " << backlink[i] << std::endl;
    }

    // STARTING POINT, I.E. LAST BEAT.. PICK A STRONG POINT IN cumscore VECTOR
    d_vec_t tmp_vec;
    for (int i = df_len - beat_period[beat_period.size()-1] ; i < df_len; i++) {
        tmp_vec.push_back(cumscore[i]);
    }

    int startpoint = get_max_ind(tmp_vec) +
        df_len - beat_period[beat_period.size()-1] ;

    // can happen if no results obtained earlier (e.g. input too short)
    if (startpoint >= int(backlink.size())) {
        startpoint = int(backlink.size()) - 1;
    }

    // USE BACKLINK TO GET EACH NEW BEAT (TOWARDS THE BEGINNING OF THE FILE)
    //  BACKTRACKING FROM THE END TO THE BEGINNING.. MAKING SURE NOT TO GO BEFORE SAMPLE 0
    i_vec_t ibeats;
    ibeats.push_back(startpoint);
//    std::cerr << "startpoint = " << startpoint << std::endl;
    while (backlink[ibeats.back()] > 0) {
//        std::cerr << "backlink[" << ibeats.back() << "] = " << backlink[ibeats.back()] << std::endl;
        int b = ibeats.back();
        if (backlink[b] == b) break; // shouldn't happen... haha
        ibeats.push_back(backlink[b]);
    }

    // REVERSE SEQUENCE OF IBEATS AND STORE AS BEATS
    for (int i = 0; i < int(ibeats.size()); i++) {
        beats.push_back(double(ibeats[ibeats.size() - i - 1]));
    }
}


