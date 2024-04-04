import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record, SharedAdam
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
from column_env import RandomObstaclesEnv
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 300000

env = RandomObstaclesEnv()
N_S = env.observation_space.shape
N_A = env.action_space.n


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
        
        pi1 = torch.tanh(self.pi1(x))
        pi1 = torch.tanh(self.pi2(pi1))
        pi1 = torch.cat((pi1, p), dim=1)
        pi1 =  torch.tanh(self.pi3(pi1))
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
        return m.sample().numpy()[0]

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


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = RandomObstaclesEnv().unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()[0]
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            action_list = []
            while True:
                if self.name == 'w00':
                    pass#self.env.render()

                a = self.lnet.choose_action(v_wrap(s[None, :]))
                
                action_list.append(str(a))
                
                s_, r, done, trancated, _ = self.env.step(a)
                if done: r = 0
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        with open('action_file', 'a') as f:
    # Iterate over each element in the list
                            for item in action_list:
                                f.write(item + ' ')
                            f.write('\n')
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-6, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()