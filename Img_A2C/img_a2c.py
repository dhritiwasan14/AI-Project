import gym
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

#os.chdir("D:/School Stuff/50 .021 Artificial Intelligence/Project/Img_A2C/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE) #define Optimzer

    EXTRA = 0
    ep_rewards = []
    entropy_term = 0
    mean_rewards = []
    steps_done = 0

    if LOAD:
        # policy_net.load_state_dict(torch.load(BESTMODEL))
        checkpoint = torch.load(BESTMODEL)
        steps_done = checkpoint['steps_total'] 
        EXTRA = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])  # load model
        optimizer.load_state_dict(checkpoint['optimizer'])  # load optimiser
        if PLAY == False:
            df = pd.read_csv(BESTCSV, header=None)
            ep_rewards = np.transpose(np.array(df)).reshape(-1).tolist() #list(torch.unbind(torch.FloatTensor(np.array(df).tolist()).to(device)))
        print(f"Loading from Episode {EXTRA}. Current steps = {steps_done}") #Load message
        
    if PLAY:
        print(f"Playing 1 game of Pong, with model trained for {steps_done} steps ({EXTRA} episodes).")
        rewards = []
        state = prepro(env.reset())
        for steps in count():
            env.render() ## RENDER ##
            value, prob_dist = model.forward(state)
            value = value.item()
            dist = prob_dist.detach().squeeze(0).to('cpu').numpy()

            #sample A ~ pi (.|S, theta)
            action = np.random.choice(n_actions, p=dist)

            new_state, reward, done, _ = env.step(action) # next step
            new_state = prepro(new_state) #preprocess the new state
            rewards.append(reward)
            state = new_state

            if done:
                print(f"Game ended in {steps} steps, Reward: {np.sum(rewards)}")
                break
        
    else:
        print(f"Running Pong for {MAX_EPISODES}. Starting from Episode {EXTRA}") #declare start
    
        for episode in range(MAX_EPISODES):
            log_probs = []
            values = []
            rewards = []
            entropy_term = 0 #reset entropy
            
            state = prepro(env.reset()) #preprocess the state
            #steps_done = 0 # i want to get the total total steps completed
    
            for steps in range(MAX_STEPS):
                #env.render() ## RENDER ##
                value, prob_dist = model.forward(state)
                #value = value.item() #value.detach().to('cpu').numpy()[0, 0]
                dist = prob_dist.detach().squeeze(0).to('cpu').numpy()
    
                #sample A ~ pi (.|S, theta)
                action = np.random.choice(n_actions, p=dist)
    
                #Calculate ln ( pi(A|S, theta), entropy = - SUM(p(x) * ln(p(x))
                log_prob = torch.log(prob_dist.squeeze(0)[action])
                #entropy = calc_entropy(dist)
                entropy = -sum((prob_dist * torch.log(prob_dist)).squeeze(0)) #propogates gradients
    
                new_state, reward, done, _ = env.step(action) # next step
                new_state = prepro(new_state) #preprocess the new state
                rewards.append(reward)
                values.append(value.squeeze(0))
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
    
                if done or steps == MAX_STEPS - 1:
                    Qval, _ = model.forward(new_state)
                    Qval = Qval.item()#Qval.detach().to('cpu').numpy()[0, 0]
                    ep_rewards.append(np.sum(rewards))
                    mean_rewards.append(np.mean(ep_rewards[-100:]))
                    steps_done += steps#steps_done = steps
                    print(f"Episode: {episode+1+EXTRA}, Reward: {np.sum(rewards)} | Mean reward: {mean_rewards[-1]}")
                    #plot_durations(ep_rewards) ## PLOT TRAINING PROGRESS
                    break
    
            # Q = E [r(t+1) + GAMMA* V(s (t+1)]
            Qvals = np.zeros_like(rewards)#(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval
    
            # update actor critic
            values = torch.cat(values)#torch.FloatTensor(values).to(device)
            Qvals = torch.FloatTensor(Qvals).to(device)
            log_probs = torch.stack(log_probs)
    
            #Advantage = Q (s, a) - V (s)
            advantage = Qvals - values.detach()
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = F.smooth_l1_loss(Qvals, values) #0.5 * advantage.pow(2).mean()
                
            ac_loss = actor_loss + critic_loss - ALPHA * entropy_term
    
            optimizer.zero_grad()
            ac_loss.backward()
            optimizer.step()
            #plt.ioff()
            #plt.show()
            if ((episode+1+EXTRA) % 50 == 0 & (episode+1+EXTRA) % 200 != 0):
                plot_durations(ep_rewards) ##Show update every X episode
            if ((episode+1+EXTRA) % 10 == 0):  ### save every X episodes #################
                print(f"Saving Checkpoint")
                #Save CSV of data
                csvname = 'TillEp_' + str(episode+1+EXTRA) + '_data.csv'
                np.savetxt(os.path.join(CSV_DIR, csvname), ep_rewards, delimiter=',')
    
                # Output Result on Screen
                #avg_last_100 = np.average(np.array(torch.cat(ep_rewards).to('cpu').numpy())[-100:])
                avg_last_100 = np.average(np.array(ep_rewards)[-100:])
                print(f"Average over last 100 = {avg_last_100}")
    
                # Save model
                cp_model = 'pong_' + str(episode+1+EXTRA) + '.pth.tar'
                save_state = {
                    'steps_total': steps_done,
                    'epoch': episode+1+EXTRA,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(save_state, os.path.join(MODEL_DIR, cp_model))
                
                #Save Image
                rebuild1(CSV_DIR+csvname)
                plot_durations(ep_rewards)

def calc_entropy(dist):
    entropy = 0
    for i in dist:
        entropy -= i * np.log(i)
    return entropy
    
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
    #plt.show()
 
## Parameters   
BEST = 10    #Latest/Best Episode
LOAD = True
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

## Hyper-Parameters 
MAX_EPISODES = 5000#10000
GAMMA = 0.99
MAX_STEPS = int(2e7) #per episode
ALPHA = 0.001
LEARNING_RATE = 1e-3 #7e-4 #for adam optimiser, default 1e-3


if __name__ == "__main__":
    start_time = time.time()
    env = gym.make("Pong-v0")
    a2c(env)
    env.close()
    print(f"This run of {MAX_EPISODES} episodes took {round((time.time() -  start_time)/60,2)} minutes.")
    
    
    
    
    
