import os, sys 
sys.path.append("../")

from utils import * 
import numpy as np
import pickle


from utils import PADDING

class DataPreparation:
    
    def __init__(self, df):
        self.df = df
        
    def create_label_encoder(self):
        """
        Save and Pickle label encoder. 
        It converts family_accession labels into integers
        """
        unique_family_accession = set(self.df.family_accession.values.tolist() + ['<unknown>'])
        d = dict(zip(unique_family_accession, [i for i in range(len(unique_family_accession))]))
        pickle.dump( d, open( "label_encoder.p", "wb" ) )
    
    @staticmethod
    def encode_labels(labels):
        """
        Encode family_accession labels with pickled label encoders.
        This function also takes care of unknown family_accession that can be seen in the testing.
        All unseen family_accession in the training data will be categoried in one class.
        """
        d = pickle.load(open("label_encoder.p", "rb" ))
        lab = ['<unknown>' if s not in d.keys() else s for s in labels]
        out = list(map(lambda x : d[x], lab)) 
        return np.array(out, dtype=np.int64)
    
    def encode_sequence(self):
        """ 
        Create a new column "encoded_sequence" that encodes the column "sequence"
        Encode sequence into an array of integer.
        i.e
        
        "ACD" becomes np.array([1,2,3])
        """
        self.df['encoded_sequence'] = self.df['sequence'].apply(lambda x : encode_sequence(x))
    
    def encode_family_accession(self):
        """ 
        Create a new column "encode_family_accession" that encodes the column "family_accession"
        Encode famil into an array of integer.
        i.e
        
        "PF02953.15" becomes 1 for example
        """
        self.df['encoded_family_accession'] = self.encode_labels(self.df.family_accession)
    
    def torchable_columns(self):
        """
        Modifying data in order to feed it to Pytorch model
        """
        #please make sure that sequences are encoded
        col = self.df.encoded_sequence.values
        arr = np.empty([col.shape[0], len(col[0])])
        for i, element in enumerate(col) : 
            arr[i,:] = np.array(element)
        return arr