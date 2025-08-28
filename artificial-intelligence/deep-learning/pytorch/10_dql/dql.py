"""
deep q learning
cart pool

"""
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple,deque
from itertools import count


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython import display

env = gym.make("CartPole-v1",render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple("Transition",
                        ( "state","action","next_state","reward"))

# replay memory oluşturma
class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen = capacity)
 
    def push(self,*args):
        self.memory.append(Transition(*args))
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


# dql modeli oluşturma
class DQN(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(n_observations,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,n_actions)

    def forward(self,x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.layer3(x)


# hiperparametrelerin ve yardımcı fonksiyonların belirlenmesi
batch_size = 128
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005
lr = 1e-4

n_actions = env.action_space.n # action sayisi
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations,n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(),lr = lr)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    if state is None:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations(show_result = False):
    plt.figure(1)
    duration_t = torch.tensor(episode_durations,dtype = torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(duration_t.numpy())

    if len(duration_t) > 100:
        means = duration_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    display.display(plt.gcf())
    display.clear_output(wait=True)



def optimize_model():
    # hafıza kontrolü yeterli sayıda devam var mı yok mu
    if len(memory) < batch_size:
        return
    
    # hafızadan rastgele deneyim ornegi alınır(geçmişte öğrenilen bilgiler)
    transitions = memory.sample(batch_size)
    # ayirma işlemi
    batch = Transition(*zip(*transitions))

    # sonraki durumları none olmayan boolean maskesi oluşturur
    non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, batch.next_state)),device = device, dtype = torch.bool)
    # terminal olmayan tüm durumları tek bir tensor halinde birleştirir
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


    # durum, aksiyon ve ödüllerin birleştirilmesi
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1,action_batch)
    next_state_values = torch.zeros(batch_size, device = device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    # ödül + sonraki değer
    expected_state_action_values = (next_state_values*gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# model eğitimi ve sonucların değerlendirilmesi 
num_episodes = 250

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # durumu belirle
        action = select_action(state)
        # hareket et
        observation, reward, terminated, truncated, _ = env.step(action.item())
        # ödül
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated # kaybetti ya da yandı

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # hafızaya kaydet
        memory.push(state, action, next_state, reward)

        # sonraki adım
        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    
print("Done")
plot_durations(show_result=True)
plt.show()
 


