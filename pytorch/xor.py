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
        X = F.sigmoid(self.fc1(X))
        ''' 
        F.sigmoid() is apparently deprecated. The alternative
        is torch.sigmoid(). The arrangement is in such a way
        as to keep general math functions in torch and those
        specific to neural networks in torch.nn
        '''
        X = self.fc2(X)
        return X

def predict_xor(X,y,net):
    torch.set_printoptions(precision=20)
    print("Input")
    print(X)
    print("\n")
    print("True Output")
    print(y)
    print("\n")
    print("Predicted Output")
    print(net(X))
    print("\n")

def main():
    # XOR Input and Output
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float)
    y = torch.tensor([[0],[1],[1],[0]],dtype=torch.float)

    # Init NN
    net = SimpleFFNet()

    # Predictions Before Training
    predict_xor(X,y,net)

    # 5000 Epochs
    for ep in range(5000):
        # Forward Pass
        out = net(X)
        # Compute Loss
        loss = net.criterion(out,y)

        # BackProp
        net.zero_grad()
        loss.backward()

        learning_rate = 0.2
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    print("Training Completed!")

    # Predictions
    predict_xor(X,y,net)

if __name__ == "__main__":
    main()
