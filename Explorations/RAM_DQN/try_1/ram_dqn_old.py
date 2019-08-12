import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import os
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#os.chdir("D:/School Stuff/50 .021 Artificial Intelligence/Project/")
env = gym.make('Pong-ram-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward') )

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def load(self, stuff):
        self.memory = stuff

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

## Q-network
class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        #self.fc1 = nn.Linear(inputs,24) 
        #self.fc2 = nn.Linear(24,128) 
        #self.fc3 = nn.Linear(128,outputs)
        self.fc1 = nn.Linear(inputs,256) 
        self.fc2 = nn.Linear(256,128) 
        self.fc4 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,outputs)
        

    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc4(x))
        
        x = self.fc3(x)
        return x


## Training

def select_action(state): #e-greedy
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            output = policy_net(state)
            return torch.argmax(output).unsqueeze(0)
    else:
        return torch.tensor([random.randrange(3)], device=device, dtype=torch.long) #random 0,1,2

def plot_durations(episode_durations):
    #[tensor([-21.], device='cuda:0'), tensor([-21.], device='cuda:0')]
    if len(episode_durations) > 0:
        plt.figure(2)
        plt.clf()
        durations_t = torch.cat(episode_durations).to('cpu')#torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
    
        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

## Training Loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    try:
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s.view(1,-1) for s in batch.next_state if s is not None])
        
        state_batch = [i.view(1,-1) for i in batch.state]
        state_batch = torch.cat(state_batch,0)
    
        action_batch = torch.transpose(torch.cat(batch.action).unsqueeze(0),0,1)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # next_state_values[non_final_mask] = torch.tensor(np.max(target_net(non_final_next_states).detach().float().to('cpu').numpy(),axis=1)).to(device)#target_net(non_final_next_states).argmax(1).detach().float()
        # # Compute the expected Q values
        # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        # Compute Huber loss
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    
        # MSE Loss
        #loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Dbl Q
        ap = policy_net(non_final_next_states).argmax(1).detach().view(-1,1)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, ap).view(-1)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    except:
        pass

## Convert Action to Action states
def interpret_action(raw_action):
# take in select_action(state)
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
# so we only need 0, 2, 3
    ra = raw_action.item()
    if ra == 1:
        do = 2
    elif ra == 2:
        do = 3
    else:
        do = 0
    return torch.tensor(np.array(do),dtype = torch.int64).unsqueeze(0).to(device)

## Main Loop
def runloop(num_episodes):
    print(f"Running Pong for {num_episodes}. Starting from Episode {EXTRA}")
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        # env.reset() returns a np array of size 128 (RAM State)
        state = torch.from_numpy(env.reset()).float().to(device)
        episode_reward = 0
        for t in count():
            #if (i_episode+EXTRA+1) % 100 == 0:
                #env.render() ##RENDER##
            # Select and perform an action
            action = interpret_action(select_action(state))
            obs, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
    
            # Observe new state
            if not done:
                next_state = torch.from_numpy(obs).float().to(device)
            else:
                next_state = torch.from_numpy(env.reset()).float().to(device) #reset if done(?)
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
            episode_reward += reward
    
            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                plot_durations(reward_list) ##DISPLAY REWARDS FOR NOW##
                break
        reward_list.append(episode_reward)
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
        print(f"Episode {i_episode+1+EXTRA}: {np.array(torch.cat(reward_list).to('cpu').numpy())[-1]} | Average:{np.average(np.array(torch.cat(reward_list).to('cpu').numpy())[-100:])}")
        
        if (i_episode+EXTRA+1) % 100 == 0: #save every X steps
            print(f"Saving Checkpoint")
            #Save CSV of data
            csvname = 'TillEp_' +str(i_episode+1+EXTRA) + '_data.csv'
            np.savetxt(os.path.join(CSV_DIR, csvname), reward_list, delimiter=',')
            

            #Output Result on Screen
            avg_last_100 = np.average(np.array(torch.cat(reward_list).to('cpu').numpy())[-100:])
            print(f"Average over last 100 = {avg_last_100}")
            
            #Save model
            cp_model = 'pong_'+str(i_episode+1+EXTRA)+'.pth.tar'
            save_state = {
                        'steps_total' : steps_done,
                        'epoch': i_episode+1+EXTRA,
                        'policy_state_dict': policy_net.state_dict(),
                        'target_state_dict': target_net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'replay_mem': memory.memory,
                    }
            torch.save(save_state, os.path.join(MODEL_DIR, cp_model))
            #cp_model = 'pong_'+str(i_episode+1+EXTRA)+'.pt'
            #torch.save(policy_net.state_dict(),os.path.join(MODEL_DIR, cp_model))
    #plt.close() 
    return reward_list

## Params
EXTRA = 0
SAVE_DIR = "./RAM"
CSV_DIR = "./RAM/csv"
MODEL_DIR = "./RAM/model"
LOAD = True
BESTCSV = "./RAM/csv/TillEp_2600_data.csv"
BEST = "./RAM/model/pong_2600.pth.tar"

    
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 30#10
LEARNING_RATE = 2e-2 # betwee 1e-4 to 1e-2
num_episodes = 1400#500
memory = ReplayMemory(50000)

input_space = env.observation_space.shape[0]#128
n_actions = 3 #env.action_space.n #6 but need to convert to 3

policy_net = (DQN(input_space,n_actions).to(device)).float()
target_net = (DQN(input_space,n_actions).to(device)).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, betas = (0.9, 0.999))

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

steps_done = 0
reward_list = []

if LOAD:
    #policy_net.load_state_dict(torch.load(BEST))
    checkpoint = torch.load(BEST)
    steps_done = checkpoint['steps_total']
    EXTRA = checkpoint['epoch']
    print(EXTRA)
    policy_net.load_state_dict(checkpoint['policy_state_dict']) #load policy net
    target_net.load_state_dict(checkpoint['target_state_dict']) #load target net
    optimizer.load_state_dict(checkpoint['optimizer']) #load optimiser
    memory.load(checkpoint['replay_mem']) #load memory
    df = pd.read_csv(BESTCSV, header=None)
    reward_list = list(torch.unbind(torch.FloatTensor(np.array(df).tolist()).to(device)))

start_time = time.time()
runloop(num_episodes)
print(f"This run of {num_episodes} episodes took {round((time.time() -  start_time)/60,2)} minutes.")

#RAM State 
# array([192,   0,   0,   0, 110,  38,   0,   7,  71,   1,  60,  59,   0,
#          0,   0,  62, 255,   0, 255, 253,   0,  22,   0,  24, 128,  32,
#          1,  86, 247,  86, 247,  86, 247, 134, 243, 245, 243, 240, 240,
#        242, 242,  32,  32,  64,  64,  64, 188,  65, 189,   0,  22, 109,
#         37,  37,  60,   0,   0,   0,   0, 109, 109,  37,  37, 192, 192,
#        192, 192,   1, 192, 202, 247, 202, 247, 202, 247, 202, 247,   0,
#          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#          0,   0,   0,   0,   0,   0,  54, 236, 242, 121, 240], dtype=uint8)

# env.observation_space.shape[0]

# env.action_space.n #6
# env.get_action_meanings()
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
# so we only need 0, 2, 3



