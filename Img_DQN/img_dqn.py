#!/usr/bin/env python3
import os
from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

#from tensorboardX import SummaryWriter
#os.chdir("D:/School Stuff/50 .021 Artificial Intelligence/Project/Img_DQN/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_ENV_NAME = "Pong-v0"
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
BEST = 10    #Latest/Best Episode
LOAD = False
PLAY = False
# Load: False + Play: False = Fresh Init Training
# Load:  True + Play: False = Pretrained Training
# Load: False + Play:  True = Untrained Game
# Load:  True + Play:  True = Test (Trained) Game


CSV_DIR = "./csv/"                            #Folder for CSVs
MODEL_DIR = "./model/"                        #Folder for Save states
IMG_DIR = "./images/"                         #Folder for Graphs
BESTCSV = f"{CSV_DIR}TillEp_{BEST}_data.csv"     
BESTMODEL = f"{MODEL_DIR}pong_{BEST}.pth.tar" 


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
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
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device=device):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).type(torch.LongTensor).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)
    
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)
  
def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

def rebuild1(csvfile):
    plt.close()
    df = pd.read_csv(csvfile, header=None)
    npdf = np.array(df).reshape(-1)
    plt.figure(2)
    npdf_t = torch.tensor(npdf, dtype=torch.float)
    avg_last_100 = np.average(npdf[-100:])
    plt.title('Average over last 100 = ' + str(avg_last_100))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(npdf)
    # Take 100 episode averages and plot them too
    if len(npdf_t) >= 100:
        means = npdf_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(os.path.join(IMG_DIR,f'{npdf.shape[0]}.png'))
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args(args=[])
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    best_mean_reward = None
    episodes = 0
    EXTRA = 0
    
    if LOAD:
        checkpoint = torch.load(BESTMODEL)
        frame_idx = checkpoint['steps_total'] 
        EXTRA = checkpoint['epoch']
        net.load_state_dict(checkpoint['net_state_dict'])  # load net model
        tgt_net.load_state_dict(checkpoint['tgt_state_dict'])  # load tgt net model
        optimizer.load_state_dict(checkpoint['optimizer'])  # load optimiser
        if PLAY == False:
            df = pd.read_csv(BESTCSV, header=None)
            total_rewards = np.transpose(np.array(df)).reshape(-1).tolist() #list(torch.unbind(torch.FloatTensor(np.array(df).tolist()).to(device)))
        print(f"Loading from Episode {EXTRA}. Current steps = {frame_idx}") #Load message
    
    if PLAY:
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
        
    else:
        print(f"Running Pong (DQN_Image) for {MAX_EPISODES}. Starting from Episode {EXTRA}") #declare start
        while True:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    
            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                episodes += 1
                total_rewards.append(reward)
                mean_reward = np.mean(total_rewards[-100:])
                print(f"Episode: {episodes+EXTRA}, Reward: {np.sum(reward)} | Mean reward: {mean_reward}")
    
                    
                if ((episodes+EXTRA) % 200 == 0):  ### save every X episodes #################
                    print(f"Saving Checkpoint")
                    #Save CSV of data
                    csvname = 'TillEp_' + str(episodes+EXTRA) + '_data.csv'
                    np.savetxt(os.path.join(CSV_DIR, csvname), total_rewards, delimiter=',')
        
                    # Output Result on Screen
                    avg_last_100 = np.average(np.array(total_rewards)[-100:])
                    print(f"Average over last 100 = {avg_last_100}")
        
                    # Save model
                    cp_model = 'pong_' + str(episodes+EXTRA) + '.pth.tar'
                    save_state = {
                        'steps_total': frame_idx,
                        'epoch': episodes+EXTRA,
                        'net_state_dict': net.state_dict(),
                        'tgt_state_dict': tgt_net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        #'replay_mem': buffer.buffer #too big....
                    }
                    torch.save(save_state, os.path.join(MODEL_DIR, cp_model))
                    
                    #Save Image
                    rebuild1(CSV_DIR+csvname)    
                
                if episodes == MAX_EPISODES:
                    break
    
            if len(buffer) < REPLAY_START_SIZE:
                continue
    
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
    
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()


