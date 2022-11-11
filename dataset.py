import os

import numpy as np
import pandas as pd
import scipy

DIR_RAW = 'Data/raw_data'
DIR_FILTERED = 'Data/filtered_data'
DIR_ICA_FILTERED = 'Data/ica_filtered_data'

LABELS_PATH = 'Data/scales.xls'

COLUMNS_TO_RENAME = {
    'Subject No.': 'subject_no',
    'Trial_1': 't1_math',
    'Unnamed: 2': 't1_mirror',
    'Unnamed: 3': 't1_stroop',
    'Trial_2': 't2_math',
    'Unnamed: 5': 't2_mirror',
    'Unnamed: 6': 't2_stroop',
    'Trial_3': 't3_math',
    'Unnamed: 8': 't3_mirror',
    'Unnamed: 9': 't3_stroop'
}

TEST_TYPES = ["Arithmetic", "Mirror", "Stroop"]

TEST_TYPE_COLUMNS = {
    'Arithmetic': ['t1_math', 't2_math', 't3_math'],
    'Mirror': ['t1_mirror', 't2_mirror', 't3_mirror'],
    'Stroop': ['t1_stroop', 't2_stroop', 't3_stroop']
}

DATA_TYPES = ["raw", "filtered", "ica_filtered"]

def load_dataset(data_type="ica_filtered", test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset with the test specified by test_type.
    The data_type parameter specifies which of the datasets to load. Possible values
    are raw, filtered, ica_filtered.
    Returns a Numpy Array with shape (120, 32, 3200).
    '''
    assert (test_type in TEST_TYPES)

    assert (data_type in DATA_TYPES)

    if data_type == "ica_filtered" and test_type != "Arithmetic":
        print("Data of type", data_type, "does not have test type", test_type)
        return 0

    if data_type == "raw":
        dir = DIR_RAW
        data_key = 'Data'
    elif data_type == "filtered":
        dir = DIR_FILTERED
        data_key = 'Clean_data'
    else:
        dir = DIR_ICA_FILTERED
        data_key = 'Clean_data'
        
    dataset = np.empty((120, 32, 3200))

    counter = 0
    for filename in os.listdir(dir):
        if test_type not in filename:
            continue

        f = os.path.join(dir, filename)
        data = scipy.io.loadmat(f)[data_key]
        dataset[counter] = data
        counter += 1
    return dataset


def load_labels():
    '''
    Loads labels from the dataset and transforms the label values to binary values.
    Values larger than 5 are set to 1 and values lower than or equal to 5 are set to zero.
    '''
    labels = pd.read_excel(LABELS_PATH)
    labels = labels.rename(columns=COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > 5
    return labels


def format_labels(labels, test_type="Arithmetic", epochs=1):
    '''
    Filter the labels to keep the labels from the test type specified by test_type.
    Repeat the labels by the amount of epochs in a recording, specified by epochs.
    '''
    assert (test_type in TEST_TYPES)

    formatted_labels = []
    for trial in TEST_TYPE_COLUMNS[test_type]:
        formatted_labels.append(labels[trial])

    formatted_labels = pd.concat(formatted_labels).to_numpy()

    formatted_labels = formatted_labels.repeat(epochs)

    return formatted_labels


def convert_to_epochs(dataset, n_channels, sfreq):
    '''
    Splits EEG data into epochs with length 1 sec.
    '''
    epoched_dataset = np.empty(
        (dataset.shape[0], dataset.shape[2]//sfreq, n_channels, sfreq))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[2]//sfreq):
            epoched_dataset[i, j] = dataset[i, :, j*sfreq:(j+1)*sfreq]
    return epoched_dataset
