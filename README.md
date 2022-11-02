# EEG Stress Detection
Classification of stress using EEG recordings from the SAM 40 dataset. A description of the dataset can be found [here](https://www.sciencedirect.com/science/article/pii/S2352340921010465).

## Files
The code is split into Jupyter notebooks.

**dataset**

Contains functions for loading and transforming the dataset

```load_dataset(raw=True, test_type="Arithmetic")```
Loads data from the SAM 40 Dataset with the test specified by test_type.
The raw flag specifies whether to use the raw data or the filtered data.
Returns a Numpy Array with shape (120, 32, 3200).

```load_labels()```

Loads labels from the dataset and transforms the label values to binary values.
Values larger than 5 are set to 1 and values lower than or equal to 5 are set to zero.

```convert_to_epochs(dataset, channels, sfreq)```

Splits EEG data into epochs with length 1 sec


**filtering**

An experimental notebook focused on filtering the data and removing artifacts through linear filters and ICA.

The filtering is performed using the ```mne``` package which is a Python package specialised in MEG and EEG analysis and visualisation.

**features**

```time_series_features(data, channels)```
Generate the features mean, variance, skewness and rms using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials*n_secs, n_channels*n_features)

```nonlinear_features(data, channels)```
Compute the features Hurst exponent, Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials*n_secs, n_channels*n_features)

```entropy_features(data, channels, sfreq)```
 Compute the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials*n_secs, n_channels*n_features)

```hjorth_features(data, channels, sfreq)```
Compute the features Hjorth mobility (spectral), Hjorth complexity (spectral), Hjorth mobility and Hjorth complexity using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials*n_secs, n_channels*n_features)

```freq_band_features(data, channels, sfreq, freq_bands)```
Compute the frequency bands delta, theta, alpha, beta and gamma using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials*n_secs, n_channels*n_features)


**classification**

Classification using features loaded from **features.py**. Uses an LR classifer, KNN classifier, SVM classifier and an MLP.