import mne
import mne_features
import numpy as np

def time_series_features(data, channels):
    stats_to_compute = 4
    n_features = channels* stats_to_compute

    features = np.empty([data.shape[0], data.shape[1], n_features])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            mean = mne_features.univariate.compute_mean(second)
            variance = mne_features.univariate.compute_variance(second)
            skewness = mne_features.univariate.compute_skewness(second)
            rms = mne_features.univariate.compute_rms(second)
            stats = np.concatenate([mean, variance, skewness, rms])
            features[i][j] = stats
    features = features.reshape([features.shape[0]*features.shape[1], features.shape[2]])
    return features

def nonlinear_features(data, channels):
    stats_to_compute = 3
    n_features = channels* stats_to_compute

    features = np.empty([data.shape[0], data.shape[1], n_features])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            hurst = mne_features.univariate.compute_hurst_exp(second)
            higuchi = mne_features.univariate.compute_higuchi_fd(second)
            katz = mne_features.univariate.compute_katz_fd(second)
            svd_entropy = mne_features.univariate.compute_svd_entropy(second)
            entropies = np.concatenate([hurst, higuchi, katz])
            features[i][j] = entropies
    features = features.reshape([features.shape[0]*features.shape[1], features.shape[2]])
    return features

def entropy_features(data, channels, sfreq):
    stats_to_compute = 4
    n_features = channels* stats_to_compute

    features = np.empty([data.shape[0], data.shape[1], n_features])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            app_entropy = mne_features.univariate.compute_app_entropy(second)
            samp_entropy = mne_features.univariate.compute_samp_entropy(second)
            spect_entropy = mne_features.univariate.compute_spect_entropy(sfreq, second)
            svd_entropy = mne_features.univariate.compute_svd_entropy(second)
            entropies = np.concatenate([app_entropy, samp_entropy, spect_entropy, svd_entropy])
            features[i][j] = entropies
    features = features.reshape([features.shape[0]*features.shape[1], features.shape[2]])
    return features

def hjorth_features(data, channels, sfreq):
    stats_to_compute = 4
    n_features = channels* stats_to_compute

    features = np.empty([data.shape[0], data.shape[1], n_features])
    for i, trial in enumerate(data):
        for j, second in enumerate(trial):
            mobility_spect = mne_features.univariate.compute_hjorth_mobility_spect(sfreq, second)
            complexity_spect = mne_features.univariate.compute_hjorth_complexity_spect(sfreq, second)
            mobility = mne_features.univariate.compute_hjorth_mobility(second)
            complexity = mne_features.univariate.compute_hjorth_complexity(second)
            hjorth = np.concatenate([mobility_spect, complexity_spect, mobility, complexity])
            features[i][j] = hjorth
    features = features.reshape([features.shape[0]*features.shape[1], features.shape[2]])
    return features

def freq_band_features(data, channels, sfreq, freq_bands):
    features = np.empty([0, channels*(len(freq_bands)-1)])
    for trial in data:
        PSDh2 = np.empty([0, channels*(len(freq_bands)-1)])
        for second in trial:
            PSD_delta = mne_features.univariate.compute_pow_freq_bands(
                sfreq, second, freq_bands=freq_bands, normalize=True, ratios=None, ratios_triu=False, psd_method='welch', log=False, psd_params=None)
            PSDh2 = np.vstack((PSDh2, PSD_delta))
        features = np.vstack((features, PSDh2))
    return features
