import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NN2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim ,out_channels1, out_channels2, hidden_size, linear_hidden, nb_classes):
        super(NN2,self).__init__()
        #parameters for embedding layer
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        #parameter for conv layers
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2

        #parameter for bilstm 
        self.hidden_size = hidden_size
        
        #parameter for linear layers 
        self.linear_hidden = linear_hidden
        
        # parameter of dataset 
        self.nb_classes = nb_classes

        #Embedding layer
        # output shape is (*, H) where H=embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

        #conv layer 1
        self.conv1 = nn.Conv2d(1, self.out_channels1, (4, 4), stride=2)
        self.maxpool1 = nn.MaxPool2d(3, stride=1)
        
        #conv2 layer 2
        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, 2, stride=2)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        #bilstm layer 
        self.bilstm = nn.LSTM(input_size=500, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        
        #linear layer
        #64 * self.out_channels2 * 273 * 3 + 64 * self.embedding_dim * 2 * self.hidden_size
        self.linear1 = nn.Linear(1131 , self.linear_hidden)
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(self.linear_hidden, self.nb_classes)

    def forward(self, x, h0, c0):
        # shape of x is : (*, 2200)
        
        # applying embedding layer
        x = self.embedding(x)
        x0 = x.permute(0,2,1)
        # shape of x is : (*, 2200, H) where H=self.embedding_dim
        # adding a new dimension to feed it to conv layer 
        x = x.unsqueeze(dim=1)
        # shape of x is : (*, 1, 2200, H) where H=self.embedding_dim
        
        #applying conv layer 1 
        x = self.conv1(x) 
        #shape of x is : (*, self.out_channels1, H, W) where H = [2200 - (4 -1) -1]/2 + 1  = 1099 and W = [self.embedding_dim - (4-1)-1]/2 + 1 = self.embedding_dim/2 - 1
        x = torch.relu(x)
        x = self.maxpool1(x)
        #shape of x is : (*, self.out_channels1, H, W) where H = [1099 - (3-1)-1]/1 +1 = 1097 and W = [self.embedding_dim/2 -1 -(3-1) -1] + 1 = self.embedding_dim/2 -3

        # Convolution layer 2 is applied
        x = self.conv2(x)
        #shape of x is : (*, self.out_channels2, H, W) where H = (1097 - (2-1)-1)/2 + 1 = floor(548,5) and W = [self.embedding_dim/2 -3 -(3-1)-1)/2 + 1 = self.embedding_dim/2 - 4
        x = torch.relu(x)
        x = self.maxpool2(x)
        #shape of x is : (*, self.out_channels2, H, W) where H = (548 + 2*0 - (3-1)-1)/2 + 1 = floor(273.5) and W = (self.embedding_dim/2 - 4 + 2*0 - (3-1)-1)/2 + 1 = self.embedding_dim/4 -2,5
        
        #bilstm layer input data for layer needs to be 
        # x of shape (seq_len, batch, input_size) here my seq_len is the length of the padded encoded protein sequence ie 2200
        # h_0 of shape (num_layers * num_directions, batch, hidden_size) and c_0 of shape (num_layers * num_directions, batch, hidden_size)
        x1, _ = self.bilstm(x0, (h0, c0))
        x1 = torch.relu((x1))

        #h_n ie x1 of shape (num_layers * num_directions, batch, hidden_size) = (2, self.num_embedding, self.hidden_size)
        
        x = nn.Flatten()(x)
        #size of x is self.out_channels2 * 1096 * (self.embedding_dim/2 - 4) 

        x1 =  nn.Flatten()(x1)
        #x1 shape is (seq_len, batch, num_directions * hidden_size) ie (self.embedding_dim , 64, 2*self.hidden_size)
        
        # concat size of 2 * 64 * self.hidden_size + 64 * self.out_channels2 * 1096 * (self.embedding_dim/2 - 4)
        concat = torch.cat((x, x1), -1)
        out = F.relu(self.linear1(concat))
        out = self.linear2(out)
        return out