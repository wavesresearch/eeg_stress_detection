# EEG Stress Detection
Classification of stress using EEG recordings from the SAM 40 dataset. A description of the dataset can be found [here](https://www.sciencedirect.com/science/article/pii/S2352340921010465).

## Files

**dataset**

Contains functions for loading and transforming the dataset

```load_dataset(data_type="ica_filtered", test_type="Arithmetic")```

Loads data from the SAM 40 Dataset with the test specified by test_type.
The data_type parameter specifies which of the datasets to load. Possible values are raw, wt_filtered, ica_filtered.
Returns an ndarray with shape (120, 32, 3200).

```load_labels()```

Loads labels from the dataset and transforms the label values to binary values.
Values larger than 5 are set to 1 and values lower than or equal to 5 are set to zero.

```format_labels(labels, test_type="Arithmetic", epochs=1)```

Filter the labels to keep the labels from the test type specified by test_type.
Repeat the labels by the amount of epochs in a recording, specified by epochs.


```split_data(dataset, sfreq)```

Splits EEG data into epochs with length 1 sec.


**filtering**

A notebook for filtering data using bandpass filtering, Savitzky-Golay filtering and ICA filtering.

ICA components can be removed using visual inspection of the components to determine the ones corresponding to noise and artifacts, and selection can be performed using a GUI.

The data can be saved to a directory to be used for classification.

The filtering is performed using the [```mne``` package](https://mne.tools/stable/index.html) which is a Python package specialised in MEG and EEG analysis and visualisation.

**features**

```time_series_features(data)```

Compute the features peak-to-peak amplitude, variance and rms using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials\*n_secs, n_channels\*n_features)

```freq_band_features(data, freq_bands)```

Compute the frequency bands delta, theta, alpha, beta and gamma using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials\*n_secs, n_channels\*n_features)

```hjorth_features(data)```

Compute the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials\*n_secs, n_channels\*n_features)

```fractal_features(data)```

Compute the Higuchi Fractal Dimension and Katz Fractal Dimension using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials\*n_secs, n_channels\*n_features)

```entropy_features(data)```

Compute the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.
The data should be on the form (n_trials, n_secs, n_channels, sfreq)
The output is on the form (n_trials\*n_secs, n_channels\*n_features)

**classification**

Classification using features loaded from **features**. Uses a KNN classifier, SVM classifier and an MLP to classify.

**variables**

Script containing the global variables used in the project.