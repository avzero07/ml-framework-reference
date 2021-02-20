import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Reference Program that Trains a Simple Feedforward Network
to learn XOR.
'''

class SimpleFFNet(nn.Module):
    
    def __init__(self):
        super(SimpleFFNet, self).__init__()
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

def predict_xor(X,y,net):
    
    op = net(X)

    for i in range(4):
        print("Input #{}\t\t= [{:.1f},{:.1f}]".format(i,X.data[i][0],X.data[i][1]))
        print("True Output\t\t= [{:.1f}]".format(y.data[i][0]))
        print("Predicted Output\t= [{:.1f}]".format(op.data[i][0]))
        print("\n")

def main():
    print("\nXOR Test Program Start\n")
    # XOR Input and Output
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float)
    y = torch.tensor([[0],[1],[1],[0]],dtype=torch.float)

    # Init NN
    net = SimpleFFNet()

    # Predictions Before Training
    print("Predictions Using Untrained Net")
    print_hbar()
    predict_xor(X,y,net)

    # Train for 5000 Epochs
    epoch_count = 5000
    learning_rate = 0.2
    print("Training on FeedForward Network over {} Epochs".format(epoch_count))
    loss = net.train(X,y,epoch_count,learning_rate)
    print("Training Completed: Final Loss = {:.2f}%\n".format(loss*100))

    # Predictions
    print("Predictions Using Trained Net")
    print_hbar()
    predict_xor(X,y,net)

# Helper Functions

def print_hbar(len=25):
    print("="*len)

if __name__ == "__main__":
    main()
