import warnings
import copy

import numpy as np
import numba as nb
from numba.types import bool_
from scipy.optimize import minimize


WARN_KWARGS = {
    'category': RuntimeWarning,
    'message': 'SVD optimal rank is 0. The largest singular values are '
                'indistinguishable from noise. Setting rank truncation to 1.',
    'module': r'pydmd',
    'lineno': 73
}


@nb.njit
def first_false(arr, reverse=False, default=-1):
    step = -1 if reverse else 1
    start = (arr.shape[0] - 1) if reverse else 0
    stop = -1 if reverse else arr.shape[0]
    for idx in range(start, stop, step):
        if arr[idx] == False:
            return idx
    return default


@nb.njit
def discard_small_corr(corr, Z, perturbation_thresh=0.001):
    reverse = np.abs(Z) < 1
    if reverse:
        corr = corr[::-1]

    for i in range(1, corr.shape[0]):
        perturbation_distance = np.abs(corr[i] - corr[i-1]) if not (np.isnan(corr[i]) or np.isnan(corr[i-1])) else 0
        if perturbation_distance > perturbation_thresh:
            corr[:i] = np.nan
            break
        if i == corr.shape[0] - 1:
            corr[:] = np.nan
    return corr[::-1] if reverse else corr


@nb.njit
def calculate_dynamics(Z, A, duration, offset):
    return (np.power(
        np.repeat(Z, duration),
        (np.arange(duration) - offset)
    ) * A)


@nb.njit
def calculate_corr(signal, candidate_dynamics):
    corr_denom = np.std(signal) * np.std(candidate_dynamics) * signal.shape[0]
    return signal * candidate_dynamics / corr_denom


@nb.njit
def expand_window_midpoint(corr, window_midpoint, corr_thresh=-0.002):
    nan_mask = np.isnan(corr)
    outer_nan_mask = np.zeros(corr.shape, dtype=bool_)
    outer_nan_mask[
        first_false(nan_mask, default=0):
        first_false(nan_mask, reverse=True, default=-2) + 1
    ] = True
    
    corr_mask = ((corr > corr_thresh) | nan_mask) & outer_nan_mask
    start = first_false(
        corr_mask[:window_midpoint],
        reverse=True, default=-2
    ) + 1
    end = first_false(
        corr_mask[window_midpoint:],
        default=corr.shape[0] - window_midpoint
    ) + window_midpoint
    
    if np.all(nan_mask[start:end]):
        return -1, -1
    
    start += first_false(nan_mask[start:]) // 2
    end -= (end - first_false(nan_mask[:end], reverse=True) - 1) // 2
    
    return start, end


@nb.njit
def expand_window_edges(corr, start, end, corr_thresh=-0.002):
    nan_mask = np.isnan(corr)
    outer_nan_mask = np.zeros(corr.shape, dtype=bool_)
    outer_nan_mask[
        first_false(nan_mask, default=0):
        first_false(nan_mask, reverse=True, default=-2) + 1
    ] = True
    
    corr_mask = ((corr > corr_thresh) | nan_mask) & outer_nan_mask
    
    start = first_false(
        corr_mask[:start],
        reverse=True, default=start
    ) + 1
    end = first_false(
        corr_mask[end:],
        default=corr.shape[0] - end
    ) + end
    
    return start, end


def optimize_rotation(
    window_centered_signal, candidate_dynamics, corr_range_start, corr_range_end
):
    return minimize(
        lambda x: -np.sum(
            calculate_corr(
                window_centered_signal[corr_range_start:corr_range_end],
                (
                    1j*np.sin(x[0]) + np.cos(x[0])
                ) * candidate_dynamics[corr_range_start:corr_range_end]
            ).real
        ),
        [0],
        bounds=[(-np.pi/2, np.pi/2)]
    ).x[0]


@nb.njit
def get_candidateZ_corr_data(
    window_centered_signal, window_len, Z, Xi, t0, perturbation_thresh=0.001,
    corr_thresh=-0.002, min_wavelength_match=0.5, wav_mag_thresh=0
):
    candidate_dynamics = calculate_dynamics(
        Z,
        Xi,
        window_centered_signal.shape[0],
        t0
    )
    candidate_real = candidate_dynamics.real
    candidate_exponential = calculate_dynamics(
        np.abs(Z),
        np.abs(Xi),
        window_centered_signal.shape[0],
        t0
    ).real

    corr = calculate_corr(window_centered_signal, candidate_real)
    corr[
        np.abs(candidate_real) < (candidate_exponential / (np.sqrt(2)*3))
    ] = np.nan
    corr = discard_small_corr(corr, Z, perturbation_thresh=perturbation_thresh)

    corr_range_start, corr_range_end = expand_window_midpoint(
        corr,
        t0 + (window_len // 2),
        corr_thresh=corr_thresh
    )

    if corr_range_start == -1 and corr_range_end == -1:
        return (
            -1, # start
            -1, # end
            0, # len
            0, # sum_corr
            np.array([np.nan]), # corr
            1
        )

    period = 1 / (np.log(Z).imag / (2*np.pi))
    envelope_peak = np.amax(
        candidate_exponential[corr_range_start:corr_range_end]
    )
    if (
        ((period * min_wavelength_match) > (corr_range_end - corr_range_start))
        or (envelope_peak < wav_mag_thresh)
    ):
        return (
            -1, # start
            -1, # end
            0, # len
            0, # sum_corr
            np.array([np.nan]), # corr
            1
        )
    
    # Xi still contains some numerical inaccuracy in the phase and,
    # by proxy, the amplitude. Can optimize from -pi/2 to pi/2.
    rot = 0.
    with nb.objmode(rot='float64'):
        rot += optimize_rotation(
            window_centered_signal,
            candidate_dynamics,
            corr_range_start,
            corr_range_end
        )
#         print(rot)
    
    Xi *= (1j*np.sin(rot) + np.cos(rot))
    candidate_dynamics *= (1j*np.sin(rot) + np.cos(rot))
    candidate_real = candidate_dynamics.real
    
    _corr = calculate_corr(window_centered_signal, candidate_real)
    _corr = discard_small_corr(_corr, Z, perturbation_thresh=perturbation_thresh)
    _corr[
        np.isnan(
            discard_small_corr(
                window_centered_signal,
                0.9,
                perturbation_thresh=perturbation_thresh
            )
        ) | np.isnan(
            discard_small_corr(
                window_centered_signal,
                1.1,
                perturbation_thresh=perturbation_thresh
            )
        )
    ] = np.nan

    corr_range_start, corr_range_end = expand_window_edges(
        _corr,
        corr_range_start,
        corr_range_end,
        corr_thresh=corr_thresh
    )

    return (
        corr_range_start,
        corr_range_end,
        corr_range_end - corr_range_start, # len
        np.nansum(np.sqrt(corr[corr_range_start:corr_range_end])), # sum_corr
        corr, # corr
        Xi # new Xi
    )


class ZDecomposition:

    
    def __init__(self, HDMD_obj, optimal_rank_tol=0):
        self.HDMD_obj = HDMD_obj
        self.optimal_rank_tol = optimal_rank_tol


    def get_hankel_candidate_Zs(
        self, signal, window_len, min_freq=0, max_freq=np.inf,
        perturbation_thresh=0.001, corr_thresh=-0.002, min_wavelength_match=0.5,
        wav_mag_thresh=0
    ):
        warnings.filterwarnings("error", **WARN_KWARGS)
        candidate_Zs = []
        noisy_window_count = 0
        for i in range(signal.shape[0] - window_len):
            try:
                signal_window = signal[i:i+window_len]
                # TODO: replace median/mean with something more akin to a mode
#                 offset = np.mean(signal_window)
                centered_signal_window = signal_window.copy()# - offset
                window_centered_signal = signal.copy()# - offset

                try:
                    hdmd = copy.deepcopy(self.HDMD_obj)
                    hdmd.fit(centered_signal_window)
                except RuntimeWarning as w:
                    if w.tau - np.amax(w.s) > self.optimal_rank_tol:
                        raise w
                    warnings.filterwarnings("ignore", **WARN_KWARGS)
                    hdmd = copy.deepcopy(self.HDMD_obj)
                    hdmd.fit(centered_signal_window)
                    warnings.filterwarnings("error", **WARN_KWARGS)

                freqs = np.log(hdmd.eigs).imag / (2*np.pi)
                dynamics_idx = np.argwhere(
                    (freqs > min_freq)
                    & (freqs < max_freq)
                    & (np.abs(hdmd.amplitudes) > 0)
                ).flatten()

                for d_idx in dynamics_idx:
                    Z = hdmd.eigs[d_idx]
                    Xi = np.mean(
                        hdmd.modes[:,d_idx]
                        / np.power(
                            np.repeat(hdmd.eigs[d_idx], hdmd.d),
                            np.arange(hdmd.d)[::-1]
                        )
                    ) * hdmd.amplitudes[d_idx]

                    corr_data = get_candidateZ_corr_data(
                        window_centered_signal,
                        window_len,
                        Z,
                        Xi,
                        i,
                        perturbation_thresh=perturbation_thresh,
                        corr_thresh=corr_thresh,
                        min_wavelength_match=min_wavelength_match,
                        wav_mag_thresh=wav_mag_thresh
                    )
                    
                    candidate_Zs.append({
                        'Z': Z,
                        'A': hdmd.amplitudes[d_idx],
                        'M': hdmd.modes[:,d_idx],
                        'Xi': corr_data[5],
                        't0': i,
#                         'offset': offset,
                        'start': corr_data[0],
                        'end': corr_data[1],
                        'len': corr_data[2],
                        'sum_corr': corr_data[3],
                        'corr': corr_data[4]
                    })
            except RuntimeWarning as w:
                noisy_window_count += 1
        return candidate_Zs, noisy_window_count
    
    def sift(
        self, signal, window_len, min_freq=0, max_freq=np.inf,
        perturbation_thresh=0.001, corr_thresh=-0.002, min_wavelength_match=0.5,
        wav_mag_thresh=0
    ):
        wav_mag_thresh = np.std(signal) * 0.5
        candidate_Zs, noisy_window_count = self.get_hankel_candidate_Zs(
            signal, window_len, min_freq=min_freq, max_freq=max_freq,
            perturbation_thresh=perturbation_thresh, corr_thresh=corr_thresh,
            min_wavelength_match=min_wavelength_match, wav_mag_thresh=wav_mag_thresh,
        )

        sum_corrs = np.array([candidate_Z['sum_corr'] for candidate_Z in candidate_Zs])
        best_Z_idx = np.argmax(sum_corrs)

        Z = candidate_Zs[best_Z_idx]
        
        if Z['len'] == 0:
            warnings.warn("No optimal Z found.")
            return Z, signal

        Z_dynamics = calculate_dynamics(
            Z['Z'],
            Z['Xi'],
            signal.shape[0],
            Z['t0']
        ).real
        
        scale = (
            np.nanmean(signal[Z['start']:Z['end']]**2)
            / np.nanmean(Z_dynamics[Z['start']:Z['end']]**2)
        )**0.5
        print(scale)
        
        Z['Xi'] *= scale

        signal[Z['start']:Z['end']] -= (Z_dynamics[Z['start']:Z['end']] * scale)

        return Z, signal