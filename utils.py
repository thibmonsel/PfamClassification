import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import random
import os

TRAIN = '../random_split/train/'
amino_acid_code = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S','T','W', 'Y', 'V', 'O', 'U', 'B', 'Z', 'X', 'J']
amino_acid_encoder = dict(zip(amino_acid_code, np.arange(1, len(amino_acid_code)+1, dtype=np.int64)))
PADDING = 2200 #since max length sequence in train is 2200 for sequence (refer to pd profiler)

def open_data(filename) : 
    """
    Opens csv file and returns a df
    """
    return pd.read_csv(filename)

def concatenate_data(save_filename):
    """
    Concatenate csv files from the train data direction in '../random_split/'.
    """
    df_init = pd.DataFrame(columns=['family_id', 'sequence_name', 'family_accession', 'aligned_sequence', 'sequence'])
    for filename in os.listdir(TRAIN):
        df = open_data(TRAIN + filename)
        frames = [df_init, df]
        df_init = pd.concat(frames)
    df_init.to_csv(save_filename)
 
def get_information_on_data(filename):
    """
    Gets general information from filename file.
    """
    df = open_data(filename)
    print('------------------')
    print("File information {}".format(filename))
    print("File has {} elements".format(df.shape[0]))
    for col in df.columns : 
        print("There are {} unique {} and here is an example : \n {} \n --------------".format(len(df[col].unique()), col, df[col][random.randint(0, df.shape[0])] ))

def generate_html_profiling(filename,report_name):
    """
    Generate a panda profiling report for a filename file and save it as a report_name html file.
    """
    df = open_data(filename)
    profile =  ProfileReport(df, title='Profiling Report of {}'.format(filename), explorative=True)
    profile.to_file(report_name)

def encode_sequence(sequence):
    """
    Encode a protein sequence into integers
    """
    tmp = list(map(amino_acid_encoder.get, list(sequence)))
    tmp += [0] * (PADDING -len(tmp))
    return tmp

def from_prediction_get_family_accession(n):
    """
    From the output prediction given by model get the family accession name of n
    """
    d = pickle.load(open("label_encoder.p", "rb" ))