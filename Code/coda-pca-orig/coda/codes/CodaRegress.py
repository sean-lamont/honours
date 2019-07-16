	#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from collections import OrderedDict
import numpy as np

class CoDA_Regress(nn.Module):

    def __init__(self, input_dim, dimension, encoder_shape, decoder_shape):
        super(CoDA_Regress, self).__init__()

        #define regression layer
        self.linear = nn.Linear(dimension, 1)

        encoder_dict = OrderedDict()

        #first layer will be twice input size, since we are feeding in both c_kl and X
        encoder_dict["layer0"] = nn.Linear(2 * input_dim, encoder_shape[0])

        for i in range(0,len(encoder_shape)-1):
            encoder_dict["layer"  + str(i)] = nn.Linear(encoder_shape[i], encoder_shape[i+1])
            encoder_dict["layer_ac"  + str(i)] = nn.ELU()

        encoder_dict["final_layer"] = nn.Linear(encoder_shape[-1], dimension)
        encoder_dict["final_ac"] = nn.ELU()

        self.encoder = nn.Sequential(encoder_dict)

        decoder_dict = OrderedDict()
        decoder_dict["layer0"] = nn.Linear(dimension, decoder_shape[0])

        for i in range(0,len(decoder_shape)-1):
            decoder_dict["layer"  + str(i)] = nn.Linear(decoder_shape[i], decoder_shape[i+1])
            decoder_dict["layer_ac"  + str(i)] = nn.ELU()

        #final layer will map back to input dim
        decoder_dict["final_layer"] = nn.Linear(decoder_shape[-1], input_dim)
        decoder_dict["final_ac"] = nn.ELU()

        self.decoder = nn.Sequential(decoder_dict)


    def forward(self,x):
        EPS = 1e-6   # to avoid log(0)

        #run the encoding and store the low level representation as A
        x_ckl = torch.log(torch.clamp(check(x), EPS, 1))

        #pass in both x and x_ckl as per paper
        A = self.encoder(torch.cat((x, x_ckl), 1))
        reconstruction = self.decoder(A)
        pred = self.linear(A)
        #inlude the predicted target and the reconstruction so both can be inputs to the combined loss
        return pred, reconstruction, A


    def fit(self, X, y, lam, lr):

        loss_function = Combined_Loss(lam)
        optim = torch.optim.Adam(self.parameters(), lr = lr)


        prev_loss = np.inf

        for epoch in range(0,5000):
            pred, recon, A = self.forward(torch.FloatTensor(X))
            loss = loss_function(recon, torch.FloatTensor(X), pred, torch.FloatTensor(y))

            optim.zero_grad()

            loss.backward()
            optim.step()

            epoch += 1


            if (epoch % 100 == 0):
                print("epoch {}, loss {}".format(epoch, loss))
        return

    def transform(self, X):
        pred, recon, A = self.forward(X)
        return A

    def predict(self, X):
        pred, recon, A = self.forward(X)
        return pred

    #recon remains in CLR space, since the loss is derived for similarity to x_ckl
    def project(self, X):
        pred, recon, A = self.forward(X)
        return recon

class Combined_Loss(torch.nn.Module):
    def __init__(self, lam):
        super(Combined_Loss,self).__init__()
        self.CoDA_Loss = CoDA_Loss()
        self.MSE = nn.MSELoss()
        self.lam = lam

    def forward(self,Y,X,y_hat,y):
        #X is original data, Y is CoDA reconstruction, y is targets, y_hat
        #input needs to be normalised by g(x) (geometric mean) for X_hat
        #TODO centering matrix? Reduce mean? Mask for near zero values?

        #extract reconstruction and original data from concatenation
        return  self.MSE(y_hat, y) + self.lam * self.CoDA_Loss(Y,X)

class CoDA_Loss(torch.nn.Module):

    def __init__(self):
        super(CoDA_Loss,self).__init__()

    def forward(self,Y,X):
        #X is original data, Y is CoDA reconstruction
        X_check = check(X)
        coda_loss =  torch.sum(torch.exp(torch.clamp(Y, -30, 30))) - torch.sum(X_check * Y)
        return coda_loss

def check(X):
    #assume input is tensor so we can use the numpy() method
    assert type(X) == torch.Tensor
    gmean = torch.prod(X, 1) ** (1./X.shape[1])
    return torch.div(X.t(), torch.clamp(gmean, min=1e-8)).t()
