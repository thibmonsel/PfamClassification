import argparse
import os, sys 
from utils import * 

import os, sys 
sys.path.append("./classify")
from data_preparation import DataPreparation
from network import NN2
  
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--sequence', type=str, help='protein sequence')
args = parser.parse_args()

NUM_EMBEDDINGS = len(amino_acid_code) + 1
EMBEDDING_DIM = 36
OUT_CHANNELS1 = 7
OUT_CHANNELS2 = 5
HIDDEN_SIZE = 3
LINEAR_HIDDEN = 9000
NUM_CLASSES = len(pickle.load(open("./classify/label_encoder.p", "rb" ))) #given the label encoder  


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        model = NN2(NUM_EMBEDDINGS, EMBEDDING_DIM ,OUT_CHANNELS1 ,OUT_CHANNELS2, HIDDEN_SIZE, LINEAR_HIDDEN, NUM_CLASSES).to(device)
        model.load_state_dict(torch.load('./classify/network.pth', map_location={'cuda:0': 'cpu'}))
        model.to(device)
        model.eval()
        x = torch.from_numpy(np.array(encode_sequence(args.sequence), dtype=np.float64)).long()
        x = x.unsqueeze(dim=0)
        x = Variable(x).to(device)
        h0,c0 = torch.randn(1*2, 1, HIDDEN_SIZE).to(device).float(), torch.randn(1*2, 1, HIDDEN_SIZE).to(device).float()
        out = model(x,h0,c0)
        prediction = out.argmax(dim=1, keepdim=True) 
        encoding = pickle.load(open("./classify/label_encoder.p", "rb" ))
        re_encoding = {v: k for k, v in encoding.items()}
        print("For the sequence {}... we predict that its family accession is {}.".format(args.sequence[:20], re_encoding[prediction.item()])) 
