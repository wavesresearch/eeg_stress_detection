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

DATA_TYPES = ["raw", "wt_filtered", "ica_filtered"]

TEST_TYPES = ["Arithmetic", "Mirror", "Stroop"]

TEST_TYPE_COLUMNS = {
    'Arithmetic': ['t1_math', 't2_math', 't3_math'],
    'Mirror': ['t1_mirror', 't2_mirror', 't3_mirror'],
    'Stroop': ['t1_stroop', 't2_stroop', 't3_stroop']
}

N_CLASSES = 2
SFREQ = 128