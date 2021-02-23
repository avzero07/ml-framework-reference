import os
import sys
sys.path.insert(0,os.path.join('..','util'))
import read_idx as rd

sys.path.insert(0,os.path.join('..','util','test'))
from test_read_idx import gunzip_to_dir
sys.path.insert(0,os.path.join('..','data','mnist'))
from mnist_dataset import MNISTDataset, MismatchedDataError

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
        X = self.conv2(X)
        X = F.relu(X)
        X = F.max_pool2d(X,2)
        X = self.dropout1(X)
        X = torch.flatten(X,1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.dropout2(X)
        X = self.fc2(X)
        output = F.log_softmax(X,dim=1)
        return output

    def train_net(self,dataset_loader,epoch_count,device="cuda"):
        self.train()
        for batch_idx, sample_dict in enumerate(dataset_loader):
            data = sample_dict['image']
            target = sample_dict['label']
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_count, batch_idx * len(data), len(dataset_loader.dataset),
                    100. * batch_idx / len(dataset_loader), loss.item()))

    def validate_net(self,dataset_loader,device="cuda"):
        # Make Network Ready for Test
        self.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for sample_dict in dataset_loader:
                data = sample_dict['image']
                target = sample_dict['label']
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataset_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(dataset_loader.dataset),
                100. * correct / len(dataset_loader.dataset)))

'''
=================
Helper Functions
=================
'''
def print_tensor_details(tensor,descr):
    print("{}\t; Storage:{}\t; Shape({})".format(descr,tensor.device,tensor.size()))

def print_h_bar(count):
    print("="*count)

'''
Extract a GZ Datastore, read contents
and returns a dataset object.
'''
def load_dataset(image_file_name,label_file_name,file_path):
    
    with tempfile.TemporaryDirectory() as dirpath:
        image_file_to_read_full_path = gunzip_to_dir(image_file_name,file_path,dirpath)
        label_file_to_read_full_path = gunzip_to_dir(label_file_name,file_path,dirpath)
    
        dataset = MNISTDataset(image_file_to_read_full_path,label_file_to_read_full_path)

    return dataset

'''
=====
Main
=====
'''

def main():

    # Load Data
    data_path = os.path.join("..","data","mnist")
    train_dataset = load_dataset("train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz",data_path)
    test_dataset = load_dataset("t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz",data_path)

    # Use data loader with
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1000)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1000)

    # Init Net
    net = CNNNet()

    # Move Network to GPU
    if torch.cuda.is_available():
        device = 'cuda'
        net = net.to(device)
    else:
        device = 'cpu'

    # Initial Predictions on Test Data
    print_h_bar(20)
    # Start Training
    net.validate_net(test_loader,device)
    for epoch in range(10):
        net.train_net(train_loader,epoch,device)
        net.validate_net(test_loader,device)
    # Final Predictions on Test Data
    print_h_bar(20)

if __name__ == "__main__":
    main()
