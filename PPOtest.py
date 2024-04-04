from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import multiprocessing
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
from column_env import RandomObstaclesEnv
from  torchrl.envs.libs.gym import GymEnv, GymWrapper 
gym_env = RandomObstaclesEnv() 
base_env = GymWrapper(gym_env)
#base_env = GymEnv("InvertedDoublePendulum-v4", device=device)
env = TransformedEnv(base_env,StepCounter())
#TransformedEnv(base_env,Compose(ObservationNorm(in_keys=["observation"]),StepCounter(),),)
#env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
#print("normalization constant shape:", env.transform[0].loc.shape)
#print("observation_spec:", env.observation_spec)
#print("reward_spec:", env.reward_spec)
#print("input_spec:", env.input_spec)
#print("action_spec (as defined by input_spec):", env.action_spec)
#check_env_specs(env)
rollout = env.rollout(10) 
print("rollout of three steps:", rollout)
#rollout is a dictionary. There are at most n steps (less if done earlier) 
# The keys are: 'action' (n*6 one-hot), 'done'(n*1), 'next'(n), 'observation'(n*4*30*30),'step_count'(n), 'terminated'(n*1)
#print("rollout of three steps:", rollout)
#print("Shape of the rollout TensorDict:", rollout.batch_size)
a1=rollout['action']
action0 = torch.argmax(a1, dim=1)
a2=rollout['done']
a3=rollout['next']
a4=rollout['observation']
obs_0 = a4[0,3,:,:]
b0=a4[:,3,0,1]
b1=a4[:,3,0,0]
a5=rollout['step_count']
a6=rollout['terminated'].view(-1)
check_env_specs(env)
import torch.nn.functional as F
class Actor(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        #n_observations = n_observations.view(n_observations.shape[0]*n_observations.shape[1]*n_observations.shape[2])        
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128+2, 128)
        self.layer3 = nn.Linear(128, n_actions*2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input):
        if len(input.shape)==3:
            input = input.reshape(1,4,30,30)
        
        p = input[:,3,0,0:2].flip(0)
        x = input[:,0:3,:,:].flatten(-3)

        x = F.relu(self.layer1(x))
        x = torch.cat((x, p),dim=1)
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x))
        norm = NormalParamExtractor()
        return norm.forward(x)
    
class Value(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Value, self).__init__()
        #n_observations = n_observations.view(n_observations.shape[0]*n_observations.shape[1]*n_observations.shape[2])        
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128+2, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input):
        if len(input.shape)==3:
            input = input.reshape(1,4,30,30)
        p = input[:,3,0,0:2].flip(0)
        x = input[:,0:3,:,:].flatten(-3)
        print(x.shape)
        x = F.relu(self.layer1(x))
        x = torch.cat((x, p),dim=1)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
actor_net = Actor(3*30*30,6)
value_net = Value(3*30*30,6)
random_tensor = torch.randn(1, 4, 30, 30)
actor_net(random_tensor)
'''
actor_net = nn.Sequential(
    nn.Flatten(0,2),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

value_net = nn.Sequential(
    nn.Flatten(0,2),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)
'''

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)
policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": 0,
        "max": 1,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)


value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)
logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        a=tensordict_data['observation']
        b=tensordict_data['next']
        c=tensordict_data['action']
        d=tensordict_data['done']
        #e=tensordict_data['reward']
        f=tensordict_data['done']
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()