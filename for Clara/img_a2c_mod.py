import gym
from gym import wrappers
import numpy as np
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time

device = torch.device("cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector Andrej code"""
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        
        self.critic1 = nn.Linear(num_inputs, 100)
        self.critic2 = nn.Linear(100, 1)

        self.actor1 = nn.Linear(num_inputs, 100)
        self.actor2 = nn.Linear(100, num_actions)

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(device))
        value = F.relu(self.critic1(state))
        value = self.critic2(value)

        prob_dist = F.relu(self.actor1(state))
        prob_dist = F.softmax(self.actor2(prob_dist), dim=1)

        return value, prob_dist

def a2c(env):
    n_inputs = 6400 #input space
    n_actions = env.action_space.n  #output space

    model = ActorCritic(n_inputs, n_actions).to(device)   #define NN

    checkpoint = torch.load(BESTMODEL, map_location='cpu') # Load Model from checkpoint
    steps_done = checkpoint['steps_total'] 
    EXTRA = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])  # load model
    print(f"Loading from Episode {EXTRA}. Current steps = {steps_done}") #Load message
    
    print(f"Playing 1 game of Pong, with model trained for {steps_done} steps ({EXTRA} episodes).")
    rewards = []
    state = prepro(env.reset())
    for steps in count():
        env.render() ## RENDER ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        value, prob_dist = model.forward(state)
        value = value.item()
        dist = prob_dist.detach().squeeze(0).to('cpu').numpy()

        action = np.random.choice(n_actions, p=dist) #sample A ~ pi (.|S, theta)
        new_state, reward, done, _ = env.step(action) # next step
        new_state = prepro(new_state) #preprocess the new state
        rewards.append(reward)
        state = new_state

        if done:
            print(f"Game ended in {steps} steps, Reward: {np.sum(rewards)}")
            break
    
 
## Parameters   
BEST = 7000    #Latest/Best Episode

MODEL_DIR = "./model/"                        #Folder for Save states   
BESTMODEL = f"{MODEL_DIR}pong_{BEST}.pth.tar" 


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    a2c(env)

    #RENDER VIDEO
    env_to_wrap = wrappers.Monitor(env, '.', force = True)
    observation = env_to_wrap.reset()

    env_to_wrap.close()
    env.close()
    
    
    
    
    
