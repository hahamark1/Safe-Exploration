from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from constants_gridwolrd import *

class DQN_ff(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DQN_ff, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.to(device=device, dtype=torch.float)
        x = x.view(x.size(0), 1, -1)
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out.view(x.size(0), -1)

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DQN(nn.Module):

    def __init__(self, h=7, w=7, outputs=4, embedding=False):
        super(DQN, self).__init__()
        self.embeddings_size = EMBEDDING_SIZE
        if embedding:
            self.conv1 = nn.Conv2d(EMBEDDING_SIZE, 8, kernel_size=3, stride=2)
        else:
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        # self.bn2 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.to(device=device, dtype=torch.float)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.head(x.view(x.size(0), -1))


class SimpleCNN(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self, embedding=False):
        super(SimpleCNN, self).__init__()
        self.input_size = 7
        self.hidden_channels = 3
        self.hidden_fc = 16
        self.output_size = 4
        self.embeddings_size = EMBEDDING_SIZE
        self.embedding = embedding
        # Input channels = 3, output channels = 18
        if embedding:
            self.conv1 = torch.nn.Conv2d(self.embeddings_size, self.hidden_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = torch.nn.Conv2d(1, self.hidden_channels, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.hidden_channels * 3 * 3, self.hidden_fc)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(self.hidden_fc, self.output_size)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        if not self.embedding:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, self.hidden_channels * 3 * 3)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return (x)

class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(49, num_hidden).type(torch.FloatTensor)
        self.l2 = nn.Linear(num_hidden, num_actions)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.l1(x)
        output = self.tanh(output)
        output = self.l2(output)
        return output

class QNetwork_deeper(nn.Module):

    def __init__(self, num_hidden=[64, 128, 64]):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(49, num_hidden[0]).type(torch.FloatTensor)
        self.l2 = nn.Linear(num_hidden[0], num_hidden[1]).type(torch.FloatTensor)
        self.l3 = nn.Linear(num_hidden[1], num_hidden[2]).type(torch.FloatTensor)
        self.l4 = nn.Linear(num_hidden[2], num_actions)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.l1(x)
        output = self.tanh(output)
        output = self.l2(output)
        output = self.tanh(output)
        output = self.l3(output)
        output = self.tanh(output)
        output = self.l4(output)
        # output = self.tanh(output)
        return output