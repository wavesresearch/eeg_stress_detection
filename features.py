import mne_features.univariate as mne_f
import numpy as np


def time_series_features(data):
    '''
    Compute the features variance, RMS and peak-to-peak amplitude using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*3)
    '''
    n_trials, n_secs, n_channels, _ = data.shape
    features_per_channel = 3

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            variance = mne_f.compute_variance(second)
            rms = mne_f.compute_rms(second)
            ptp_amp = mne_f.compute_ptp_amp(second)
            features[i][j] = np.concatenate([variance, rms, ptp_amp])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def fractal_features(data):
    '''
    Compute the Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*2)
    '''
    n_trials, n_secs, n_channels, _ = data.shape
    features_per_channel = 2

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            higuchi = mne_f.compute_higuchi_fd(second)
            katz = mne_f.compute_katz_fd(second)
            features[i][j] = np.concatenate([higuchi, katz])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def entropy_features(data):
    '''
    Compute the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*4)
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = 4

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            app_entropy = mne_f.compute_app_entropy(second)
            samp_entropy = mne_f.compute_samp_entropy(second)
            spect_entropy = mne_f.compute_spect_entropy(sfreq, second)
            svd_entropy = mne_f.compute_svd_entropy(second)
            features[i][j] = np.concatenate(
                [app_entropy, samp_entropy, spect_entropy, svd_entropy])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def hjorth_features(data):
    '''
    Compute the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*2)
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = 2

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            mobility_spect = mne_f.compute_hjorth_mobility_spect(sfreq, second)
            complexity_spect = mne_f.compute_hjorth_complexity_spect(
                sfreq, second)
            features[i][j] = np.concatenate([mobility_spect, complexity_spect])
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features


def freq_band_features(data, freq_bands):
    '''
    Compute the frequency bands delta, theta, alpha, beta and gamma using the package mne_features.
    The data should be on the form (n_trials, n_secs, n_channels, sfreq)
    The output is on the form (n_trials*n_secs, n_channels*5)
    '''
    n_trials, n_secs, n_channels, sfreq = data.shape
    features_per_channel = len(freq_bands)-1

    features = np.empty([n_trials, n_secs, n_channels * features_per_channel])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            psd = mne_f.compute_pow_freq_bands(
                sfreq, second, freq_bands=freq_bands)
            features[i][j] = psd
    features = features.reshape(
        [n_trials*n_secs, n_channels*features_per_channel])
    return features
