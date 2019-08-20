import torch
from torch import nn
from collections import OrderedDict
import numpy as np
import os

class CoDA_Cl(nn.Module):

    def __init__(self, input_dim, dimension, n_classes, encoder_shape, decoder_shape):
        super(CoDA_Cl, self).__init__()

        #define linear layer
        self.linear = nn.Linear(dimension, n_classes)

        #define log softmax layer
        self.logsoft = nn.LogSoftmax(dim=1)

        encoder_dict = OrderedDict()

        #first layer will be twice input size, since we are feeding in both c_kl and X
        encoder_dict["layer0"] = nn.Linear(2 * input_dim, encoder_shape[0])

        for i in range(0,len(encoder_shape)-1):
            encoder_dict["layer"  + str(i+1)] = nn.Linear(encoder_shape[i], encoder_shape[i+1])
            encoder_dict["layer_ac"  + str(i+1)] = nn.ELU()

        encoder_dict["final_layer"] = nn.Linear(encoder_shape[-1], dimension)
        encoder_dict["final_ac"] = nn.ELU()

        self.encoder = nn.Sequential(encoder_dict)

        decoder_dict = OrderedDict()
        decoder_dict["layer0"] = nn.Linear(dimension, decoder_shape[0])

        for i in range(0,len(decoder_shape)-1):
            decoder_dict["layer"  + str(i+1)] = nn.Linear(decoder_shape[i], decoder_shape[i+1])
            decoder_dict["layer_ac"  + str(i+1)] = nn.ELU()

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

        #returns the log probabilities
        pred = self.logsoft(self.linear(A))
        #inlude the predicted target and the reconstruction so both can be inputs to the combined loss
        return pred, reconstruction, A


    def fit(self, X, y, lam, lr,  train_size, epochs = 10000):
        PATH = os.getcwd()+"model weights"
        loss_function = Combined_Loss(lam)
        optim = torch.optim.Adam(self.parameters(), lr = lr, weight_decay=0.05)

        X_train = X[:train_size]
        X_val = X[train_size:]

        y_train = y[:train_size]
        y_val = y[train_size:]

        training_loss_arr = []
        val_loss_arr = []

        prev_loss = np.inf
        best_val_loss = np.inf
        cur_val_loss = 0
        for epoch in range(0,epochs):

            pred, recon, A = self.forward(torch.FloatTensor(X_train))

            loss = loss_function(recon, torch.FloatTensor(X_train), pred, torch.LongTensor(y_train))

            optim.zero_grad()

            loss.backward()
            optim.step()

            if (len(X_val) > 0 ):
                val_pred, val_recon, val_A = self.forward(torch.FloatTensor(X_val))

                val_loss = loss_function(val_recon, torch.FloatTensor(X_val), val_pred, torch.LongTensor(y_val))

                curr_val_loss = val_loss.detach().numpy()

                if (epoch % 100 == 0 and epoch > 1000):
                    training_loss_arr.append(loss.detach().numpy())
                    val_loss_arr.append(curr_val_loss)
                #keep the best weights (maybe do this every n iterations)
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                torch.save(self.state_dict(), PATH)


            curr_loss = loss.detach().numpy()

            # if np.abs(curr_loss - prev_loss) < 1e-18 or epoch == epochs-1:
            #
            #     break

            prev_loss = curr_loss

            epoch += 1

            if (epoch % 1000 == 0):
                print("epoch {}, loss {}".format(epoch, loss))

        #self.load_state_dict(torch.load(PATH))

        return val_loss_arr, training_loss_arr

    def transform(self, X):
        pred, recon, A = self.forward(X)
        return A

    def predict(self, X):
        pred, recon, A = self.forward(X)
        #pred = pred.exp().detach()     # exp of the log prob = probability.
        #_, index = torch.max(pred,1)   # index of the class with maximum probability
        return pred

    #recon remains in CLR space, since the loss is derived for similarity to x_ckl
    def project(self, X):
        pred, recon, A = self.forward(X)
        return recon

class Combined_Loss(torch.nn.Module):
    def __init__(self, lam):
        super(Combined_Loss,self).__init__()
        self.CoDA_Loss = CoDA_Loss()
        self.NLL = nn.NLLLoss()
        self.lam = lam


    #TODO: Scale CodA Loss prior to lambda?
    def forward(self,Y,X,y_hat,y):
        return  self.NLL(y_hat, y) + self.lam * self.CoDA_Loss(Y,X)

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
