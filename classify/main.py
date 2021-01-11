import os, sys 
sys.path.append("../")
from utils import * 

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


NUM_EMBEDDINGS = len(amino_acid_code) + 1
EMBEDDING_DIM = 36
OUT_CHANNELS1 = 7
OUT_CHANNELS2 = 5
HIDDEN_SIZE = 3
LINEAR_HIDDEN = 9000
NUM_CLASSES = len(pickle.load(open("label_encoder.p", "rb" ))) #given the label encoder  

BATCH_SIZE= 64
EPOCH = 10 
learning_rate = 0.01

# we use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NN2(NUM_EMBEDDINGS, EMBEDDING_DIM ,OUT_CHANNELS1 ,OUT_CHANNELS2, HIDDEN_SIZE, LINEAR_HIDDEN, NUM_CLASSES)

#check number of parameters in model
print("Number of trainable parameters",sum(p.numel() for p in model.parameters() if p.requires_grad))

model.to(device) # puts model on GPU / CPU

#giving optimizer and loss 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

def family_accession_encoder():
    #creating family_accession label encoder 
    df = open_data('../raw_train.csv')
    prepare = DataPreparation(df)
    prepare.create_label_encoder()

def train(epoch):
    print("#### TRAINING ####")
    model.train()
    
    for filename in os.listdir('../random_split/train/'):
        
        #preprocessing raw data
        df = open_data('../random_split/train/'+filename)
        prepare = DataPreparation(df)
        prepare.encode_sequence()
        prepare.encode_family_accession()
    
        #creating dataloader for training
        train = data_utils.TensorDataset(torch.from_numpy(prepare.torchable_columns()).long(),torch.Tensor(df.encoded_family_accession.values).long())
        train_loader = data_utils.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
        print("Created train loader for file {}".format(filename))
        
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = Variable(x).to(device), Variable(target).to(device)
            h0,c0 = torch.randn(1*2, x.shape[0], HIDDEN_SIZE).to(device), torch.randn(1*2, x.shape[0], HIDDEN_SIZE).to(device)
            optimizer.zero_grad()
            out = model(x,h0,c0)
            l = loss_fn(out, target)
            l.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('batch {} [{}/{}] training loss: {}'.format(batch_idx,batch_idx*len(x),len(train_loader.dataset),l.item()))

    print("Saving model for {} epoch".format(epoch))
    torch.save(model.state_dict(), 'network.pth') 
    
       

@torch.no_grad()   
def test(epoch):
    print("#### EVALUATION ####")
    model = NN2(NUM_EMBEDDINGS, EMBEDDING_DIM ,OUT_CHANNELS1 ,OUT_CHANNELS2, HIDDEN_SIZE, LINEAR_HIDDEN, NUM_CLASSES)
    model.load_state_dict(torch.load('network.pth'))
    model.to(device)
    model.eval()
    total_correct, total_loss, dataset_length = 0, 0, 0
    for filename in os.listdir('../random_split/test/'):
        file_loss, file_correct = 0, 0  
        #preprocessing raw data
        df = open_data('../random_split/test/'+filename)
        prepare = DataPreparation(df)
        prepare.encode_sequence()
        prepare.encode_family_accession()
    
        #creating dataloader for testing
        test = data_utils.TensorDataset(torch.from_numpy(prepare.torchable_columns()).long(), torch.Tensor(df.encoded_family_accession.values).long())
        test_loader = data_utils.DataLoader(test, batch_size = BATCH_SIZE, shuffle = True)
        dataset_length += len(test_loader.dataset)
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x).to(device), Variable(target).to(device)
            h0,c0 = torch.randn(1*2, x.shape[0], HIDDEN_SIZE).to(device), torch.randn(1*2, x.shape[0], HIDDEN_SIZE).to(device)
            out = model(x,h0,c0)
            l = loss_fn(out, target)
            file_loss += l
            total_loss += l
            prediction = out.argmax(dim=1, keepdim=True) 
            file_correct += prediction.eq(target.view_as(prediction)).sum().item()
            total_correct += prediction.eq(target.view_as(prediction)).sum().item()

        taux_classif_file = 100. * file_correct / len(test_loader.dataset)
        print('For file {}, accuracy: {}%  -- testing loss {}'.format(filename, taux_classif_file, file_loss))
    taux_classif_total = 100. * total_correct / dataset_length
    print('Epoch {} : Total testing accuracy: {}%  -- testing loss {}'.format(epoch, taux_classif_file, file_loss))

if __name__ == "__main__":
    family_accession_encoder()
    for epoch in range(1, 3):
        train(epoch)
        test(epoch)
       
   
