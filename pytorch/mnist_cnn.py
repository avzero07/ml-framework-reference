import os
import sys
sys.path.insert(0,os.path.join('..','util'))
import read_idx as rd

sys.path.insert(0,os.path.join('..','util','test'))
from test_read_idx import gunzip_to_dir

import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Helper Functions

'''
Extract a GZ Datastore, read contents
and return a tensor.
'''
def process_idx_file(ip_file_name,ip_file_path,is_label=False):
    
    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(ip_file_name,ip_file_path,dirpath)
        metadata = rd.get_metadata(file_to_read_full_path)
        idx_data = rd.get_data(file_to_read_full_path,metadata)

    if is_label:
        ret_tensor = torch.tensor(idx_data,dtype=torch.long)
    else:
        ret_tensor = torch.tensor(idx_data,dtype=torch.float)
    return ret_tensor

class FFNet(nn.Module):
    
    def __init__(self):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(784,5)
        self.fc2 = nn.Linear(5,10)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.2)

    def forward(self,X):
        X = torch.tanh(self.fc1(X))
        X = torch.tanh(self.fc2(X))
        return X

    def train(self,X,y,epoch_count=10,learning_rate=0.2):
        for ep in range(epoch_count):
            # Forward Pass
            out = self(X)
            # Compute Loss
            loss = self.criterion(out,y)
            # BackProp
            self.optimizer.zero_grad() # Call the optimizer instead of net.param
            loss.backward()

            # Use Built-in Optimizerr 
            self.optimizer.step() # Does Weight Update
            print_training_update(ep,loss,epoch_interval=(epoch_count/5))

        return loss.double()

'''
=================
Helper Functions
=================
'''
def print_tensor_details(tensor,descr):
    print("{}\t; Storage:{}\t; Shape({})".format(descr,tensor.device,tensor.size()))

def print_h_bar(count):
    print("="*count)

def print_training_update(ep_count,loss,epoch_interval=5):
    if ep_count%epoch_interval == 0:
        print("Epoch {} : Loss = {:.2f}%".format(ep_count,(loss.item()*100)))

def predict_mnist(X,y,net):
    op = net(X)
    loss = net.criterion(op,y)
    print("Loss Over Test Data = {:.2f}".format(loss.item()*100))

'''
=====
Main
=====
'''

def main():

    # Load Data
    data_path = os.path.join("..","data","mnist")
    X_train = process_idx_file("train-images-idx3-ubyte.gz",data_path)
    y_train = process_idx_file("train-labels-idx1-ubyte.gz",data_path,is_label=True)
    
    X_test = process_idx_file("t10k-images-idx3-ubyte.gz",data_path)
    y_test = process_idx_file("t10k-labels-idx1-ubyte.gz",data_path,is_label=True)

    # Flatten the Image Data
    X_train = torch.flatten(X_train,start_dim=1)
    X_test = torch.flatten(X_test,start_dim=1)

    # Init Net
    net = FFNet()

    # Move Data to GPU
    if torch.cuda.is_available():
        to_gpu = True
        X_train = X_train.to('cuda')
        print_tensor_details(X_train,'X_train')
        y_train = y_train.to('cuda')
        print_tensor_details(y_train,'y_train')
        X_test = X_test.to('cuda')
        print_tensor_details(X_test,'X_test')
        y_test = y_test.to('cuda')
        print_tensor_details(y_test,'y_test')
        # Move Net to GPU
        net = net.to('cuda')

    # Initial Predictions on Test Data
    print_h_bar(20)
    predict_mnist(X_test,y_test,net)
    print_h_bar(20)

    # Start Training
    net.train(X_train,y_train,epoch_count=10000,learning_rate=0.25) 

    # Final Predictions on Test Data
    print_h_bar(20)
    predict_mnist(X_test,y_test,net)
    print_h_bar(20)

if __name__ == "__main__":
    main()
