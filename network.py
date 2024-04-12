import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record, SharedAdam

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.map_len = (s_dim[0]-1)*s_dim[1]*s_dim[2] # the length of the vector stretched by the map (the first 3 channels)
        self.pi1 = nn.Linear(self.map_len, 128)
        self.pi2 = nn.Linear(128,128)
        
        self.pi3 = nn.Linear(128+2, 128) # concatenate vector of position
        self.pi4 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(self.map_len, 128)
        self.v2 = nn.Linear(128,128)
        self.v3 = nn.Linear(128+2, 128)
        self.v4 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, input):
        if len(input.shape)==3:
            input = input.reshape(1,4,30,30) # for single input, batch = 1
        p = input[:,3,0,0:2].flip(0) #position
        x = input[:,0:3,:,:].flatten(-3) #map, stretched
        
        pi1 = torch.relu(self.pi1(x))
        pi1 = torch.relu(self.pi2(pi1))
        pi1 = torch.cat((pi1, p), dim=1)
        pi1 =  torch.relu(self.pi3(pi1))
        logits = self.pi4(pi1)
        v1 = torch.tanh(self.v1(x))
        v1 = torch.tanh(self.v2(v1))
        v1 = torch.cat((v1, p), dim=1)
        v1 = torch.tanh(self.v3(v1))
        values = self.v4(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)

        return m.sample()

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss
    