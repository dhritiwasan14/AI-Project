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
IMG_DIR = "./"

df = pd.read_csv('output A2C NOFS.txt', header = None)
reward = []
for item in df.iterrows():
    try:
        test = float(item[1][1][9:14])
        reward.append(test)
    except:
        print(item[1][1][31:])
    
npdf = np.array(reward).reshape(-1)
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