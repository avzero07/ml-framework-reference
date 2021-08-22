import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

'''
Reference Program that Trains an LSTM to predict a sine wave
'''

class SimpleLSTM(nn.Module):
    
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(1,51,batch_first=True)
        self.lstm2 = nn.LSTM(51,51,batch_first=True)
        self.linear = nn.Linear(51,1)

    def forward(self,X,hid_1_tup=None,hid_2_tup=None):
        # X --> batch x seq x feature
        if(hid_1_tup == None or hid_2_tup == None):
            op,(hid_1,cel_1) = self.lstm1(X)
            op,(hid_2,cel_2) = self.lstm2(op)
        else:
            op,(hid_1,cel_1) = self.lstm1(X,hid_1_tup)
            op,(hid_2,cel_2) = self.lstm2(op,hid_2_tup)
        op = self.linear(hid_2)
        return (op,(hid_1.detach(),cel_1.detach()),
                   (hid_2.detach(),cel_2.detach()))

class timeSeriesDataset():

    def __init__(self,x,y,seq_length,device):
        self.ip = torch.unsqueeze(x,1).double().to(device)
        self.tar = torch.unsqueeze(y,1).double().to(device)
        self.seq_length = seq_length

    def __len__(self):
        assert self.ip.shape[0] == self.tar.shape[0]
        sig_len = self.ip.shape[0]
        sig_len_usable = sig_len - self.seq_length
        return sig_len_usable

    def __getitem__(self,idx):
        if(idx<0):
            idx = len(self) + idx
        if(idx>=len(self)):
            raise DatasetError("Trying to read beyond index!")
        data = self.ip[idx:idx+self.seq_length-1,:]
        label = self.tar[idx+self.seq_length,:]
        return (data,label)

class DatasetError(Exception):

    def __init__(self,message="Dataset Error!"):
        self.message = message
        super().__init__(self.message)

def run_inference(network,data_loader,criterion):
    network.eval()
    test_losses = torch.zeros(1).to(get_device())
    for b_id,batch in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            x,y = batch[0],batch[1]
            if(b_id == 0):
                b_op,hid_1_tup,hid_2_tup = network(x)
            else:
                b_op,hid_1_tup,hid_2_tup = network(x,hid_1_tup,hid_2_tup)
            b_op = torch.squeeze(b_op,dim=0)
            b_loss = criterion(b_op,y)
            test_losses += b_loss
            if(b_id==0):
                pred = b_op
            else:
                pred = torch.cat((pred,b_op))
    total_loss = test_losses
    return (pred,total_loss)

def run_train(network,train_loader,epoch_count,optim,criterion):
    for e in tqdm(range(epoch_count)):
        network.train()
        train_losses = torch.zeros(1).to(get_device())
        for b_id,batch in enumerate(train_loader):
            network.zero_grad()
            x,y = batch[0],batch[1]
            #def closure():
            optim.zero_grad()
            if(b_id == 0):
                b_op,hid_1_tup,hid_2_tup = network(x)
            else:
                b_op,hid_1_tup,hid_2_tup = network(x,hid_1_tup,hid_2_tup)
            b_op = torch.squeeze(b_op,dim=0)
            b_loss = criterion(b_op,y)
            b_loss.backward()
            train_losses += b_loss
            #    return b_loss
            optim.step()
            #optim.step(closure)
        total_loss = train_losses
        if(e == 0):
            losses = [total_loss.item()]
        else:
            losses.append(total_loss.item())
    return losses

def main():
    print("\nSinusoid Test Program Start\n")
    # Sinusoid Input
    t = torch.from_numpy(np.linspace(0,2,1000))
    pi = torch.arccos(torch.zeros(1))*2
    sig = torch.sin(10*pi*t)
    seq_len = 100
    batch_len = 1
    criterion = torch.nn.MSELoss()
    epoch_count = 100
    learning_rate = 0.2

    X = sig[:(len(t)//2)-1]
    Y = sig[1:len(t)//2]
    train = timeSeriesDataset(X,Y,seq_len,device=get_device())

    X_test = sig[(len(t)//2)+1:-1]
    Y_test = sig[(len(t)//2)+2:]
    test = timeSeriesDataset(X_test,Y_test,seq_len,device=get_device())

    # Prep DataLoaders
    train_dl = torch.utils.data.DataLoader(train,batch_size=batch_len)
    test_dl = torch.utils.data.DataLoader(test,batch_size=batch_len)

    # Init NN
    net = SimpleLSTM().double().to(get_device())

    # Predictions Before Training
    print("Predictions Using Untrained Net")
    print_hbar()
    (pred,loss) = run_inference(net,test_dl,criterion)
    print("Test_loss (Before Training) = {}".format(loss))
    print_hbar()
    visualize_signals(Y_test.detach().cpu().numpy(),
                      pred.detach().cpu().numpy(),t,seq_len)

    # Train
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    print("Training on FeedForward Network over {} Epochs".format(epoch_count))
    losses = run_train(net,train_dl,epoch_count,optimizer,criterion)
    print("Training Completed: Final Loss = {:.2f}\n".format(losses[-1]))
    (pred,loss) = run_inference(net,train_dl,criterion)
    visualize_signals(Y.detach().cpu().numpy(),
                      pred.detach().cpu().numpy(),t,seq_len)
    print("inference loss on train : {}".format(loss))
    visualize_signals(losses)

    # Predictions
    print("Predictions Using Trained Net")
    print_hbar()
    (pred,loss) = run_inference(net,test_dl,criterion)
    print("Test_loss (After Training) = {}".format(loss))
    print_hbar()
    visualize_signals(Y_test.detach().cpu().numpy(),
                      pred.detach().cpu().numpy(),t,seq_len)

# Helper Functions

def get_device():
    if(torch.cuda.is_available()):
        device = 'cuda:{}'.format(torch.cuda.current_device())
    else:
        device = 'cpu'
    return device

def print_hbar(len=25):
    print("="*len)

def visualize_signals(sig1,sig2=None,t=None,seq_length=None):
    if(t is not None):
        time = t[(len(t)//2)+2+seq_length:]
    plt.figure()
    plt.plot(sig1,label="Ground Truth")
    if(sig2 is not None):
        plt.plot(sig2,label="Predicted")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
