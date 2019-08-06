import gym
import numpy as np
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd

EXTRA = 0
SAVE_DIR = ""
CSV_DIR = ""
MODEL_DIR = ""
LOAD = False
BESTCSV = ""
BEST = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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





def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector Andrej code"""
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


class ActorCritic1(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # self.conv1 = nn.Conv2d((80, 80),16, kernel_size = 5, stride = 2)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)


        # self.num_actions = num_actions
        self.critic1 = nn.Linear(num_inputs, 800)
        self.critic2 = nn.Linear(800, 100)
        self.critic3 = nn.Linear(100, 1)

        self.actor1 = nn.Linear(num_inputs, 800)
        self.actor2 = nn.Linear(800, 100)
        self.actor3 = nn.Linear(100, num_actions)

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0).to(device))
        value = F.leaky_relu(self.critic1(state))
        value = F.leaky_relu(self.critic2(value))
        value = self.critic3(value)

        prob_dist = F.leaky_relu(self.actor1(state))
        prob_dist = F.leaky_relu(self.actor2(prob_dist))
        prob_dist = F.softmax(self.actor3(prob_dist), dim=1)

        return value, prob_dist


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # self.conv1 = nn.Conv2d((80, 80),16, kernel_size = 5, stride = 2)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)


        # self.num_actions = num_actions
        self.critic1 = nn.Linear(num_inputs, 100)
        self.critic2 = nn.Linear(100, 1)

        self.actor1 = nn.Linear(num_inputs, 100)
        self.actor2 = nn.Linear(100, num_actions)

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic1(state))
        value = self.critic2(value)

        prob_dist = F.relu(self.actor1(state))
        prob_dist = F.softmax(self.actor2(prob_dist), dim=1)

        return value, prob_dist

MAX_EPISODES = 10000
GAMMA = 0.99
MAX_STEPS = 1000000
ALPHA = 0.001

def a2c(env):

    n_inputs = 6400
    n_actions = env.action_space.n

    model = ActorCritic(n_inputs, n_actions).to(device)
    optimizer = optim.Adam(model.parameters())

    ep_rewards = []
    entropy_term = 0
    mean_rewards = []

    if LOAD:
        # policy_net.load_state_dict(torch.load(BEST))
        checkpoint = torch.load(BEST)
        steps_done = checkpoint['steps_total']
        EXTRA = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])  # load model
        optimizer.load_state_dict(checkpoint['optimizer'])  # load optimiser
        df = pd.read_csv(BESTCSV, header=None)
        ep_rewards = list(torch.unbind(torch.FloatTensor(np.array(df).tolist()).to(device)))
        if PLAY == False:
            df = pd.read_csv(BESTCSV, header=None)
            ep_rewards = np.transpose(np.array(df)).reshape(-1).tolist() #list(torch.unbind(torch.FloatTensor(np.array(df).tolist()).to(device)))
        print(f"Loading from Episode {EXTRA}. Current steps = {steps_done}") #Load message

    if PLAY:
        print(f"Playing 1 game of Pong, with model trained for {steps_done} steps ({EXTRA} episodes).")
        rewards = []
        state = prepro(env.reset())
        for steps in count():
            env.render()  ## RENDER ##
            value, prob_dist = model.forward(state)
            value = value.item()
            dist = prob_dist.detach().squeeze(0).to('cpu').numpy()

            # sample A ~ pi (.|S, theta)
            action = np.random.choice(n_actions, p=dist)

            new_state, reward, done, _ = env.step(action)  # next step
            new_state = prepro(new_state)  # preprocess the new state
            rewards.append(reward)
            state = new_state

            if done:
                print(f"Game ended in {steps} steps, Reward: {np.sum(rewards)}")
                break

    else:
        print(f"Running Pong for {MAX_EPISODES}. Starting from Episode {EXTRA}")  # declare start

        for episode in range(MAX_EPISODES):
            log_probs = []
            values = []
            rewards = []

            state = env.reset()
            state = prepro(state)
            steps_done = 0

            for steps in range(MAX_STEPS):
                value, prob_dist = model.forward(state)
                value = value.detach().numpy()[0, 0]
                dist = prob_dist.detach().numpy()
                dist = np.squeeze(dist)

                #sample A ~ pi (.|S, theta)
                action = np.random.choice(n_actions, p=dist)

                #Calculate ln ( pi(A|S, theta), entropy = - SUM(p(x) * ln(p(x))
                log_prob = torch.log(prob_dist.squeeze(0)[action])
                entropy = calc_entropy(dist)

                new_state, reward, done, _ = env.step(action)
                new_state = prepro(new_state)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state

                if done or steps == MAX_STEPS - 1:
                    Qval, _ = model.forward(new_state)
                    Qval = Qval.detach().numpy()[0, 0]
                    ep_rewards.append(np.sum(rewards))
                    mean_rewards.append(np.mean(ep_rewards[max(episode - 100, 0):episode + 1]))
                    steps_done = steps
                    print("episode: {}, reward: {}, mean reward: {} \n".format(episode + 1 + EXTRA,np.sum(rewards), mean_rewards[-1]))
                    plot_durations(ep_rewards)
                    break

            # Q = E [r(t+1) + GAMMA* V(s (t+1)]
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval

            # update actor critic
            values = torch.FloatTensor(values).to(device)
            Qvals = torch.FloatTensor(Qvals).to(device)
            log_probs = torch.stack(log_probs)

            #Advantage = Q (s, a) - V (s)
            advantage = Qvals - values
            actor_loss = adv_actor_loss(log_probs, advantage)
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + ALPHA * entropy_term

            optimizer.zero_grad()
            ac_loss.backward()
            optimizer.step()
            plt.ioff()
            plt.show()

            if ((episode+1) % 100 == 0):  # save every X steps
                print(f"Saving Checkpoint")
                #Save CSV of data
                csvname = 'TillEp_' + str(episode + 1) + '_data.csv'
                np.savetxt(os.path.join(CSV_DIR, csvname), ep_rewards, delimiter=',')

                # Output Result on Screen
                avg_last_100 = np.average(np.array(torch.cat(ep_rewards).to('cpu').numpy())[-100:])
                print(f"Average over last 100 = {avg_last_100}")

                # Save model
                cp_model = 'pong_' + str(episode) + '.pth.tar'
                save_state = {
                    'steps_total': steps_done,
                    'epoch': episode + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(save_state, os.path.join(MODEL_DIR, cp_model))

def calc_entropy(dist):
    entropy = 0
    for i in dist:
        entropy -= i * np.log(i)
    return entropy

def adv_actor_loss(log_probs, advantage):
    return (-log_probs * advantage).mean()

def Q_actor_loss(log_probs, Qvals):
    return (-log_probs * Qvals).mean()

def TD_loss(log_probs, delta):
    return (-log_probs * delta).mean()

if __name__ == "__main__":
    env = gym.make("Pong-v0")
    print(env.unwrapped.get_action_meanings())
    a2c(env)
