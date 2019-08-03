import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.ion()
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions

        self.affine1 = nn.Linear(num_inputs, 100)
        self.affine2 = nn.Linear(100, 1)

        self.actor1 = nn.Linear(num_inputs, 100)
        self.actor2 = nn.Linear(100, num_actions)

    def forward(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        state_value = F.relu(self.affine1(x))
        state_value = self.affine2(state_value)

        probs_dist = F.relu(self.actor1(x))
        probs_dist = F.softmax(self.actor2(probs_dist), dim=1)


        return state_value, probs_dist

MAX_EPISODES = 3000
GAMMA = 0.99
num_steps = 300

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

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())



def a2c(env):

    num_inputs = 128
    num_outputs = env.action_space.n
    print(num_outputs)

    model = ActorCritic(num_inputs, num_outputs)
    ac_optimizer = optim.Adam(model.parameters(), lr = 0.001)

    all_lengths = []
    average_lengths = []
    ep_rewards = []
    entropy_term = 0


    for episode in range(MAX_EPISODES):

        log_probs = []
        rewards = []
        state_values = []

        state = env.reset()
        steps = 0
        while True:
            state_value, probs_dist = model.forward(state)

            #Selecting action randomly from dist given
            m = Categorical(probs_dist)
            action = m.sample()
            log_prob = torch.log(probs_dist.squeeze(0)[action])
            log_probs.append(log_prob)
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            state_values.append(state_value)
            state = new_state

            steps +=1
            if done:
                Qval, _ = model.forward(new_state)
                Qval = Qval.detach().numpy()[0, 0]
                print (np.sum(rewards))
                ep_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if episode % 10 == 0:
                    print("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,np.sum(rewards),steps,average_lengths[-1]))
                plot_durations(ep_rewards)
                break

        # compute Q values
        Qvals = np.zeros_like(state_values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        state_values = torch.FloatTensor(state_values)
        Qvals = torch.FloatTensor(list(Qvals))
        log_probs = torch.stack(log_probs)

        advantage = Qvals - state_values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()


if __name__ == "__main__":
    env = gym.make("Pong-ram-v0")
    print(env.unwrapped.get_action_meanings())
    a2c(env)