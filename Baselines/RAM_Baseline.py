# import necessary modules from keras
from keras.layers import Dense
from keras.models import Sequential

# creates a generic neural network architecture
model = Sequential()

# hidden layer takes a pre-processed frame as input, and has 200 units
#model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(units=200,input_dim=128, activation='relu', kernel_initializer='glorot_uniform'))

# output layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import numpy as np
import gym
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch


# reward discount used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  # we go from last reward to first one so we don't have to do exponentiations
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

def baseline(model):
  # Macros
  UP_ACTION = 2
  DOWN_ACTION = 3
  EXTRA = 0
  ep_rewards = []
  mean_rewards = []
  steps_done = 0
  
  # initialization of variables used in the main loop
  env =  gym.make('Pong-ram-v0')
  
  for episode in range(MAX_EPISODES):
    # Reinitialization
    x_train, y_train, rewards = [],[],[]
    observation = env.reset()
    prev_input = None
    
    for steps in range(MAX_STEPS):
      # preprocess the observation, set input as difference between images
      cur_input = observation
      x = cur_input - prev_input if prev_input is not None else np.zeros(128)
      prev_input = cur_input
      
      # forward the policy network and sample action according to the proba distribution
      proba = model.predict(np.expand_dims(x, axis=1).T)
      action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
      y = 1 if action == 2 else 0 # 0 and 1 are our labels
  
      # log the input and label to train later
      x_train.append(x)
      y_train.append(y)
  
      # do one step in our environment
      observation, reward, done, info = env.step(action)
      rewards.append(reward)
      
      # end of an episode
      if done or steps == MAX_STEPS - 1: 
          ep_rewards.append(np.sum(rewards))     
          mean_rewards.append(np.mean(ep_rewards[-100:]))
          steps_done += steps    
          # training
          model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=discount_rewards(rewards, gamma))
          print(f"Episode: {episode+1+EXTRA}, Reward: {np.sum(rewards)} | Mean reward: {mean_rewards[-1]}")
          break
          
    if ((episode+1+EXTRA) % 200 == 0):  ### save every X episodes #################
        print(f"Saving Checkpoint")
        #Save CSV of data
        csvname = 'TillEp_' + str(episode+1+EXTRA) + '_data.csv'
        np.savetxt(os.path.join(CSV_DIR, csvname), ep_rewards, delimiter=',')

        # Output Result on Screen
        #avg_last_100 = np.average(np.array(torch.cat(ep_rewards).to('cpu').numpy())[-100:])
        avg_last_100 = np.average(np.array(ep_rewards)[-100:])
        print(f"Average over last 100 = {avg_last_100}")
        
        #Save Image
        rebuild1(CSV_DIR+csvname)

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

                                                              

          
 
## Parameters   
#BEST = 0    #Latest/Best Episode
#LOAD = True

CSV_DIR = "./csv/"                            #Folder for CSVs
#MODEL_DIR = "./model/"                        #Folder for Save states
IMG_DIR = "./images/"                         #Folder for Graphs
#BESTCSV = f"{CSV_DIR}TillEp_{BEST}_data.csv"     
#BESTMODEL = f"{MODEL_DIR}pong_{BEST}.pth.tar"  
          


## Hyperparameters
gamma = 0.99
MAX_STEPS = int(2e7)
MAX_EPISODES = 400

baseline(model)






