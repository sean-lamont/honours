import torch
from torch import nn
from collections import OrderedDict
import numpy as np
import os

class CoDA_Regress(nn.Module):

    def __init__(self, input_dim, dimension, encoder_shape, decoder_shape):
        super(CoDA_Regress, self).__init__()

        #define regression layer
        self.linear = nn.Linear(dimension, 1)

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
        pred = self.linear(A)
        #inlude the predicted target and the reconstruction so both can be inputs to the combined loss
        return pred, reconstruction, A


    def fit(self, X, y, lam, lr,  train_size, epochs = 10000):
        PATH = os.getcwd()+"model weights"
        loss_function = Combined_Loss(lam)
        optim = torch.optim.Adam(self.parameters(), lr = lr)

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

            loss = loss_function(recon, torch.FloatTensor(X_train), pred, torch.FloatTensor(y_train))

            optim.zero_grad()

            loss.backward()
            optim.step()

            if (len(X_val) > 0 ):
                val_pred, val_recon, val_A = self.forward(torch.FloatTensor(X_val))

                val_loss = loss_function(val_recon, torch.FloatTensor(X_val), val_pred, torch.FloatTensor(y_val))

                curr_val_loss = val_loss.detach().numpy()

                if (epoch % 100 == 0):
                    training_loss_arr.append(loss.detach().numpy())
                    val_loss_arr.append(curr_val_loss)
            #     #keep the best weights (maybe do this every n iterations)
            # if cur_val_loss < best_val_loss:
            #     best_val_loss = cur_val_loss
            #     torch.save(self.state_dict(), PATH)


            curr_loss = loss.detach().numpy()

            if np.abs(curr_loss - prev_loss) < 1e-18 or epoch == epochs-1:

                # self.load_state_dict(torch.load(PATH))
                break

            prev_loss = curr_loss

            epoch += 1

            if (epoch % 1000 == 0):
                print("epoch {}, loss {}".format(epoch, loss))


        return val_loss_arr, training_loss_arr

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


    #TODO: Scale CodA Loss prior to lambda?
    def forward(self,Y,X,y_hat,y):
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



#strikes
    # def fit(self, X, y, lam, lr,  train_size, epochs = 10000,  max_strikes = 30):
    #
    #     loss_function = Combined_Loss(lam)
    #     optim = torch.optim.Adam(self.parameters(), lr = lr)
    #
    #     prev_loss = np.inf
    #
    #     X_train = X[:train_size]
    #     X_val = X[train_size:]
    #
    #     y_train = y[:train_size]
    #     y_val = y[train_size:]
    #
    #     prev_val_loss = np.inf
    #
    #     #strikes, if val loss increases max_strikes in a row, stop training and take best model so far
    #     strike_count = 0
    #
    #     training_loss_arr = []
    #     val_loss_arr = []
    #
    #     prev_loss = np.inf
    #     for epoch in range(0,epochs):
    #
    #         pred, recon, A = self.forward(torch.FloatTensor(X_train))
    #
    #         loss = loss_function(recon, torch.FloatTensor(X_train), pred, torch.FloatTensor(y_train))
    #
    #         optim.zero_grad()
    #
    #         loss.backward()
    #         optim.step()
    #
    #         if (len(X_val) > 0 ):
    #             val_pred, val_recon, val_A = self.forward(torch.FloatTensor(X_val))
    #
    #             val_loss = loss_function(val_recon, torch.FloatTensor(X_val), val_pred, torch.FloatTensor(y_val))
    #
    #             curr_val_loss = val_loss.detach().numpy()
    #
    #             if (epoch % 100 == 0):
    #                 training_loss_arr.append(loss.detach().numpy())
    #                 val_loss_arr.append(curr_val_loss)
    #
    #             if curr_val_loss > prev_val_loss:
    #                 strike_count += 1
    #             else:
    #                 strike_count = 0
    #             prev_val_loss = curr_val_loss
    #
    #         if strike_count == max_strikes:
    #             break
    #
    #         curr_loss = loss.detach().numpy()
    #
    #         if np.abs(curr_loss - prev_loss) < 1e-16:
    #             break
    #
    #         prev_loss = curr_loss
    #
    #         epoch += 1
    #
    #         if (epoch % 1000 == 0):
    #             print("epoch {}, loss {}".format(epoch, loss))
    #
    #
    #     return val_loss_arr, training_loss_arr

# Minibatch version
#     def fit(self, X, y, lam, lr, epochs = 10000, batch_size = 5):
#
#         loss_function = Combined_Loss(lam)
#         optim = torch.optim.Adam(self.parameters(), lr = lr)
#
#         prev_loss = np.inf
#
#         for epoch in range(0,epochs):
#             #X = X.detach().numpy()
#             np.random.shuffle(X)
#
#             #y = y.detach().numpy()
#             np.random.shuffle(y)
#             curr_loss = 0
#             for i in range(0, X.shape[0], batch_size):
#                 X_batch = X[i:i + batch_size]
#                 y_batch = y[i:i + batch_size]
#
#                 pred, recon, A = self.forward(torch.FloatTensor(X_batch))
#
#                 loss = loss_function(recon, torch.FloatTensor(X_batch), pred, torch.FloatTensor(y_batch))
#
#                 optim.zero_grad()
#
#                 loss.backward()
#                 optim.step()
#
#                 curr_loss += loss.detach().numpy()
#
#             if np.abs(curr_loss - prev_loss) < 1e-16:
#                 break
#
#             prev_loss = curr_loss
#             epoch += 1
#
#             if (epoch % 1000 == 0):
#                 print("epoch {}, loss {}".format(epoch, loss))
#         return
