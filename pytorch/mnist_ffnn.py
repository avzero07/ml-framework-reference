import sys
sys.path.insert(0,"../util/")
import read_idx as rd

sys.path.insert(0,"../util/test")
from test_read_idx import gunzip_to_dir

import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper Functions

'''
Extract a GZ Datastore, read contents
and return a tensor.
'''
def process_idx_file(ip_file_name,ip_file_path):
    
    with tempfile.TemporaryDirectory() as dirpath:
        file_to_read_full_path = gunzip_to_dir(ip_file_name,ip_file_path,dirpath)
        idx_data = rd.get_data(file_to_read_full_path,)

class FFNet(nn.Module):
    
    def __init__(self):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)
        self.criterion = nn.MSELoss()

    def forward(self,X):
        X = torch.sigmoid(self.fc1(X))
        ''' 
        F.sigmoid() is apparently deprecated. The alternative
        is torch.sigmoid(). The arrangement is in such a way
        as to keep general math functions in torch and those
        specific to neural networks in torch.nn
        '''
        X = self.fc2(X)
        return X

    def train(self,X,y,epoch_count=10000,learning_rate=0.01):
        for ep in range(epoch_count):
            # Forward Pass
            out = self(X)
            # Compute Loss
            loss = self.criterion(out,y)
            # BackProp
            self.zero_grad()
            loss.backward()

            for f in self.parameters():
                f.data.sub_(f.grad.data * learning_rate)

        return loss.double()

def main():
    # Load Data
    
    X_train = get_data("../")

    # Init Net
    net = FFNet()

if __name__ == "__main__":
    main()
