# EEG Stress Detection
Classification of stress using EEG recordings from the SAM 40 dataset. A description of the dataset can be found [here](https://www.sciencedirect.com/science/article/pii/S2352340921010465).

## Files
The code is split into Jupyter notebooks.

**load_dataset**
Contains functions for loading and reshaping the data.

The function ```load_dataset(raw=True, test_type="Arithmetic")``` loads either the filtered or raw dataset, which is specified by ```raw```. The argument ```test_type```specifies which of the three test types to use (Arithmetic, Mirror or Stroop).

The function ```convert_to_epochs(dataset, channels, sfreq, epoch_length=1)``` converts the data into epochs where the length is specified by ```epoch_length```.

**load_labels**
The function ```load_labels()``` loads the labels from an Excel spreadsheet and sets all stress ratings above 5 to ```True``` and the other values to ```False```.

**filtering**
An experimental file focused on filtering the data and removing artifacts through linear filters and ICA.

The filtering is performed using the ```mne``` package which is a Python package specialised in MEG and EEG analysis and visualisation.

**features_spectral**
Extracts features based on the the relative powers of the frequency bands. More information on these values can be found [here](https://www.mdpi.com/1424-8220/21/11/3786/htm).


**classification_voltage_features**
Classification using the full epoched time-series for each subject and trial. Thus, the features of the dataset are the voltages at each instance.

This notebook and the notebook **classification_spectral** use the same classifers (KNN, SVM and a neural network).

**classification_spectral**
Classification using the spectral features computed in **features_spectral**. 