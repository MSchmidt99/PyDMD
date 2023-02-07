import warnings
import copy

from .utils import compute_tlsq, compute_svd

import numpy as np
import numba as nb
from numba.types import bool_
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn', lineno=1382)


WARN_KWARGS = {
    'category': RuntimeWarning,
    'message': 'SVD optimal rank is 0. The largest singular values are '
               'indistinguishable from noise. Setting rank truncation to 1.',
    'module': r'pydmd',
    'lineno': 73
}


@nb.njit
def calculate_dynamics(Z, A, duration, offset):
    return (np.power(
        np.repeat(Z, duration),
        (np.arange(duration) - offset)
    ) * A)


def null_space(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    if num == M:
        num = -1
    Q = vh[num:,:].T.conj()
    return Q


def compute_phasor(Z, hdmd, signal, opt=True):
    # swap from calculating phasor with snapshots[x] to whole signal (opt=True)
    hdmd._sub_dmd._opt = opt
    snp, _ = hdmd._col_major_2darray(signal)
    snapshots = hdmd._pseudo_hankel_matrix(snp)
    hdmd._sub_dmd._snapshots = snapshots

    X = snapshots[:, :-1]
    Y = snapshots[:, 1:]

    sub_dmd = hdmd._sub_dmd

    X, Y = compute_tlsq(X, Y, sub_dmd.tlsq_rank)

    operator = sub_dmd.operator

    U, s, V = compute_svd(X, operator._svd_rank)

    if operator._tikhonov_regularization is not None:
        operator._norm_X = np.linalg.norm(X)
    atilde = operator._least_square_operator(U, s, V, Y)

    if operator._forward_backward:
        # b stands for "backward"
        bU, bs, bV = compute_svd(Y, svd_rank=len(s))
        atilde_back = operator._least_square_operator(bU, bs, bV, X)
        atilde = sqrtm(atilde.dot(np.linalg.inv(atilde_back)))

    if operator._rescale_mode == 'auto':
        operator._rescale_mode = s

    operator._Atilde = atilde

    if operator._rescale_mode is None:
        # scaling isn't required
        Ahat = operator._Atilde
    elif isinstance(operator._rescale_mode, np.ndarray):
        if len(operator._rescale_mode) != operator.as_numpy_array.shape[0]:
            raise ValueError('''Scaling by an invalid number of
                    coefficients''')
        scaling_factors_array = operator._rescale_mode

        factors_inv_sqrt = np.diag(np.power(scaling_factors_array, -0.5))
        factors_sqrt = np.diag(np.power(scaling_factors_array, 0.5))

        # if an index is 0, we get inf when taking the reciprocal
        for idx, item in enumerate(scaling_factors_array):
            if item == 0:
                factors_inv_sqrt[idx] = 0

        Ahat = np.linalg.multi_dot([factors_inv_sqrt, operator.as_numpy_array,
                                    factors_sqrt])
    else:
        raise ValueError('Invalid value for rescale_mode: {} of type {}'
                         .format(operator._rescale_mode,
                                 type(operator._rescale_mode)))

    operator._eigenvalues = np.array([Z])
    operator._eigenvectors = null_space(
        Ahat - Z * np.eye(Ahat.shape[0])
    )[:,0][:,np.newaxis]

    operator._compute_modes(Y, U, s, V)

    sub_dmd._svd_modes = U
    
    sub_dmd._set_initial_time_dictionary(
        {"t0": 0, "tend": snapshots.shape[1] - 1, "dt": 1}
    )

    A = sub_dmd._compute_amplitudes()[0]

    return np.mean(
        hdmd.modes
        / np.power(
            np.repeat(Z, hdmd.d),
            np.arange(hdmd.d)
        )
    ) * A


class ZDecomposition:

    
    def __init__(self, HDMD_obj, optimal_rank_tol=0):
        self.HDMD_obj = HDMD_obj
        self.optimal_rank_tol = optimal_rank_tol


    def get_hankel_candidate_Zs(
        self, signal, window_len, min_freq=0, max_freq=np.inf
    ):
        warnings.filterwarnings("error", **WARN_KWARGS)
        candidate_Zs = []
        noisy_window_count = 0
        for i in range(signal.shape[0] - window_len):
            try:
                hdmd_signal_window = signal[i:i+window_len]

                try:
                    hdmd = copy.deepcopy(self.HDMD_obj)
                    hdmd.fit(hdmd_signal_window)
                except RuntimeWarning as w:
                    if w.tau - np.amax(w.s) > self.optimal_rank_tol:
                        raise w
                    warnings.filterwarnings("ignore", **WARN_KWARGS)
                    hdmd = copy.deepcopy(self.HDMD_obj)
                    hdmd.fit(hdmd_signal_window)
                    warnings.filterwarnings("error", **WARN_KWARGS)

                with np.errstate(divide='ignore'):
                    freqs = np.log(hdmd.eigs).imag / (2*np.pi)
                
                dynamics_idx = np.argwhere(
                    (freqs > min_freq)
                    & (freqs < max_freq)
                    & (np.abs(hdmd.amplitudes) > 0)
                ).flatten()

                for d_idx in dynamics_idx:
                    Z = hdmd.eigs[d_idx]
                    if Z.imag < 0:
                        # reduce duplication through conjugates
                        Z = np.conjugate(Z)
                    
                    candidate_Zs.append({
                        'Z': Z,
                        'start': i,
                        'end': i + window_len
                    })
            except RuntimeWarning as w:
                noisy_window_count += 1
        return candidate_Zs, noisy_window_count
    
    def sift(
        self, signal, window_len, min_freq=0, max_freq=np.inf, min_windows=10,
        noise_var_tol=0.001, uniqueness_tol=1e-6
        
    ):
        candidate_Zs, noisy_window_count = self.get_hankel_candidate_Zs(
            signal, window_len, min_freq=min_freq, max_freq=max_freq
        )
        
        self.Z_spectrogram = np.array([
            # [time, real, imag]
            [Z['start'], Z['Z'].real, Z['Z'].imag]
            for Z in candidate_Zs
        ])

        # identify unique Z's
        components = np.arange(1, window_len * 2)
        bics = []
        mixtures = []
        for n_c in components:
            gm = GaussianMixture(n_components=n_c, random_state=0).fit(self.Z_spectrogram)
            bics.append(gm.bic(self.Z_spectrogram))
            mixtures.append(gm)

        self.Z_gaussians = {}
        cluster_model = mixtures[np.argmin(bics)]
        pred = cluster_model.predict(self.Z_spectrogram)
        for cluster in np.unique(pred):
            cluster_Z = self.Z_spectrogram[pred == cluster]
            if cluster_Z.shape[0] >= min_windows:
                self.Z_gaussians[cluster] = cluster_Z

        # optionally filter Z_gaussians
        for cluster in list(self.Z_gaussians.keys()):
            cluster_Z = self.Z_gaussians[cluster]
            if np.std(cluster_Z[:,1]) * np.std(cluster_Z[:,2]) > noise_var_tol:
                del self.Z_gaussians[cluster]
        
        # identify and merge duplicate Z values
        Z_values = np.array([
            np.mean(cluster_Z[:, 1]) + 1j*np.mean(cluster_Z[:, 2])
            for cluster_Z in self.Z_gaussians.values()
        ])
        clusters = list(self.Z_gaussians.keys())
        for i in range(len(Z_values)):
            if i >= len(Z_values):
                break
            sim_mask = np.abs(Z_values - Z_values[i]) < uniqueness_tol
            if sum(sim_mask) > 1:
                argwhere_sim = np.argwhere(sim_mask).flatten()
                arg_duplicates = argwhere_sim[argwhere_sim > i]
                for j in arg_duplicates[::-1]:
                    self.Z_gaussians[clusters[i]] = np.concatenate([
                        self.Z_gaussians[clusters[i]],
                        self.Z_gaussians[clusters[j]]
                    ])
                    del self.Z_gaussians[clusters[j]]
                    clusters.pop(j)
                    Z_values = np.delete(Z_values, j, 0)

        # itentify Z spans and create Z dicts
        residual = signal.copy()
        Zs = []
        for cluster, cluster_Z in self.Z_gaussians.items():
            Z = np.mean(cluster_Z[:, 1]) + 1j*np.mean(cluster_Z[:, 2])
            start = int(np.amin(cluster_Z[:, 0])) + 1
            end = int(np.amax(cluster_Z[:, 0])) + window_len
            length = end - start
            Xi = compute_phasor(
                Z,
                copy.deepcopy(self.HDMD_obj),
                signal[start:end]
            )
            dynamics = calculate_dynamics(Z, Xi, length, 0)
            scale, offset = np.linalg.lstsq(
                np.vstack([dynamics.real, np.ones(length)]).T,
                signal[start:end],
                rcond=None
            )[0]
            Xi *= scale
            dynamics *= scale
            residual[start:end] -= (dynamics.real + offset)
            Zs.append({
                'Z': Z,
                'Xi': Xi,
                'offset': offset,
                'start': start,
                'end': end,
                'len': end - start,
                'dynamics': dynamics
            })

        return Zs, residual