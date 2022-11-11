import os
import numpy as np
import pandas as pd
import scipy
import variables as v

def load_dataset(data_type="ica_filtered", test_type="Arithmetic"):
    '''
    Loads data from the SAM 40 Dataset with the test specified by test_type.
    The data_type parameter specifies which of the datasets to load. Possible values
    are raw, filtered, ica_filtered.
    Returns a Numpy Array with shape (120, 32, 3200).
    '''
    assert (test_type in v.TEST_TYPES)

    assert (data_type in v.DATA_TYPES)

    if data_type == "ica_filtered" and test_type != "Arithmetic":
        print("Data of type", data_type, "does not have test type", test_type)
        return 0

    if data_type == "raw":
        dir = v.DIR_RAW
        data_key = 'Data'
    elif data_type == "filtered":
        dir = v.DIR_FILTERED
        data_key = 'Clean_data'
    else:
        dir = v.DIR_ICA_FILTERED
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
    labels = pd.read_excel(v.LABELS_PATH)
    labels = labels.rename(columns=v.COLUMNS_TO_RENAME)
    labels = labels[1:]
    labels = labels.astype("int")
    labels = labels > 5
    return labels


def format_labels(labels, test_type="Arithmetic", epochs=1):
    '''
    Filter the labels to keep the labels from the test type specified by test_type.
    Repeat the labels by the amount of epochs in a recording, specified by epochs.
    '''
    assert (test_type in v.TEST_TYPES)

    formatted_labels = []
    for trial in v.TEST_TYPE_COLUMNS[test_type]:
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
