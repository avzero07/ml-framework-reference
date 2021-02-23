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

class CNNNet(nn.Module):
    
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128,10)
        
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adadelta(self.parameters(), lr=0.01)

    def forward(self,X):
        X = self.conv1(X)
        X = F.relu(X)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2)
        X = self.dropout1(X)
        X = torch.flatten(X,1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout2(X)
        X = self.fc2(X)
        output = F.log_softmax(X,dim=1)
        return output

    def train(self,X,y,batch_ratio=200,epoch_count=10):
        # Split to Batches
        batch_size = int(X.size()[0]/batch_ratio)
        batch_offset = 0
        for ep in range(epoch_count):
            total = 0
            correct = 0
            while(batch_offset<X.size()[0]):
                train_batch_x = X[batch_offset:batch_offset+batch_size]
                train_batch_y = y[batch_offset:batch_offset+batch_size]
                
                if torch.cuda.is_available():
                    train_batch_x = train_batch_x.to('cuda')
                    train_batch_y = train_batch_y.to('cuda')

                # Forward Pass
                out = self(train_batch_x)
                # Compute Loss
                loss = F.nll_loss(out,train_batch_y)
                # BackProp
                self.optimizer.zero_grad() # Call the optimizer instead of net.param
                loss.backward()

                # Use Built-in Optimizerr 
                self.optimizer.step() # Does Weight Update
               
                # Calculate Batch Score
                batch_score = compute_batch_score(train_batch_y,out)
                total+=batch_score[1]
                correct+=batch_score[0]
                
                batch_offset+=batch_size
                #print("batch_offset = {}".format(batch_offset))
            
            print_training_update(ep,loss,epoch_interval=(epoch_count/5))
            print("Epoch {} : Accuracy = {:.2f}%".format(ep,(100*(correct/total))))
            batch_offset = 0

    def validate(self,X,y,batch_ratio=200,epoch_count=10):
        # Split to Batches
        batch_size = int(X.size()[0]/batch_ratio)
        batch_offset = 0

        for ep in range(epoch_count):
            total = 0
            correct = 0
            with torch.no_grad():
                while(batch_offset<X.size()[0]):
                    train_batch_x = X[batch_offset:batch_offset+batch_size]
                    train_batch_y = y[batch_offset:batch_offset+batch_size]
                    
                    if torch.cuda.is_available():
                        train_batch_x = train_batch_x.to('cuda')
                        train_batch_y = train_batch_y.to('cuda')

                    # Forward Pass
                    out = self(train_batch_x)
                    # Compute Loss
                    loss = F.nll_loss(out,train_batch_y)
               
                    # Calculate Batch Score
                    batch_score = compute_batch_score(train_batch_y,out)
                    total+=batch_score[1]
                    correct+=batch_score[0]
                
                    batch_offset+=batch_size
                    #print("batch_offset = {}".format(batch_offset))
            
            print_training_update(ep,loss,epoch_interval=(epoch_count/5))
            print("Epoch {} : Accuracy = {:.2f}%".format(ep,(100*(correct/total))))
            batch_offset = 0

        return 

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
        print("Epoch {} : Loss = {:.2f}".format(ep_count,(loss.item())))

def predict_mnist(X,y,net):
    op = net.validate(X,y)
    loss = F.nll_loss(op,y)
    print("Loss Over Test Data = {:.2f}".format(loss.item()))

def compute_batch_score(y,y_pred):
    predicted = torch.max(y_pred.data,1)[0]
    correct = (predicted == y).sum().item()
    total = y.size(0)

    return correct, total

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

    # Init Net
    net = CNNNet()

    # Fix Dimensions
    '''
    Conv Layer Expects the Input to Be
    [num_samples x num_channels x row x col]
    '''
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    # Move Data to GPU
    if torch.cuda.is_available():
        to_gpu = True
        #X_train = X_train.to('cuda')
        print_tensor_details(X_train,'X_train')
        #y_train = y_train.to('cuda')
        print_tensor_details(y_train,'y_train')
        #X_test = X_test.to('cuda')
        print_tensor_details(X_test,'X_test')
        #y_test = y_test.to('cuda')
        print_tensor_details(y_test,'y_test')
        # Move Net to GPU
        net = net.to('cuda')

    # Initial Predictions on Test Data
    print_h_bar(20)
    predict_mnist(X_test,y_test,net)
    print_h_bar(20)

    # Start Training
    net.train(X_train,y_train,epoch_count=10) 

    # Final Predictions on Test Data
    print_h_bar(20)
    predict_mnist(X_test,y_test,net)
    print_h_bar(20)

if __name__ == "__main__":
    main()
