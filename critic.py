from curses.ascii import CR
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_dim, n_hidden1, n_hidden2, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hidden1)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(n_hidden2, output_dim)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
