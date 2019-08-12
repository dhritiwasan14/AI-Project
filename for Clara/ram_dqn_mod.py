import gym
from gym import wrappers
import numpy as np
import os
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

#os.chdir("D:/School Stuff/50 .021 Artificial Intelligence/Project/for Clara/")
from lib import dqn_wrappers


device = torch.device("cpu")

DEFAULT_ENV_NAME = "Pong-ram-v0"
MEAN_REWARD_BOUND = 15

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 10
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

## Parameters   
MAX_EPISODES = 600
BEST = 2800    #Latest/Best Episode
LOAD = True
PLAY = True

MODEL_DIR = "./model/"                        #Folder for Save states  
BESTMODEL = f"{MODEL_DIR}DQN_RAM_600.tar" #f"{MODEL_DIR}pong_{BEST}.pth.tar" 

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        #conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(x)

class Agent:
    def __init__(self, env):#, exp_buffer):
        self.env = env
        #self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).float().to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        #exp = Experience(self.state, action, reward, is_done, new_state)
        #self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    

def run_dqn(env):
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    agent = Agent(env)#Agent(env, buffer)
    
    checkpoint = torch.load(BESTMODEL)
    frame_idx = checkpoint['steps_total'] 
    EXTRA = checkpoint['epoch']
    net.load_state_dict(checkpoint['net_state_dict'])  # load net model
    print(f"Loading from Episode {EXTRA}. Current steps = {frame_idx}") #Load message
    
    print(f"Playing 1 game of Pong, with model trained for {frame_idx} steps ({EXTRA} episodes).")
    rewards = []
    epsilon = 0
    steps = 0
    while True:
        env.render() ## RENDER ##
        reward = agent.play_step(net, epsilon, device=device)
        steps += 1

        if reward is not None:
            print(f"Game ended in {steps} steps, Reward: {reward}")
            env.close()
            break
            
            
if __name__ == "__main__":
    #add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args(args=[])
    device = torch.device("cuda" if args.cuda else "cpu")

    #create environment
    env = gym.make(args.env)    
    
    #run
    run_dqn(env)
    env.close()
    
    
    
    
