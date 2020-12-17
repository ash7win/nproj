#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('pylab', 'inline')
import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm, trange
import os,sys
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)


# In[2]:


import numpy as np
class ConnectFour():
  def __init__(self, state=None):
    self.reset()
    if state is not None:
      self.state = state

  def reset(self):
    self.done = False
    self.state = [0]*44
    self.state[-1] = 1
    return self.state
   
  class observation_space():
    shape = (44,)

  class action_space():
    n = 42
  
  def render(self):
    print("turn %d" % self.state[-1])
    print(np.array(self.state[0:42]).reshape(6,7))
    
  def value(self, s):
    ret = 0
    a = s.copy()
    s= s.copy()
    s= np.array(s[0:42]).reshape(6,7)
    for turn in [-1, 1]:
      for c in range(4):
        for r in range(6):
           if all([x==turn for x in [s[r][c] , s[r][c+1], s[r][c+2], s[r][c+3]]]):
             ret = turn

	# Check vertical locations for win
      for c in range(7):
        for r in range(3):
          if all([x==turn for x in [s[r][c], s[r+1][c], s[r+2][c], s[r+3][c]]]):
            ret = turn

	# Check positively sloped diaganols
      for c in range(4):
        for r in range(3):
          if all([x==turn for x in [s[r][c], s[r+1][c+1], s[r+2][c+2], s[r+3][c+3]]]):
            ret = turn

	# Check negatively sloped diaganols
      for c in range(4):
        for r in range(3, 4):
          if all([x==turn for x in [s[r][c], s[r-1][c+1], s[r-2][c+2], s[r-3][c+3]]]):
            ret = turn
    return ret*a[-1]
  
  def dynamics(self, s, act):
    rew = 0
    j= -1
    if act in range(0,7):

      a = s.copy()
      s = s.copy()
      s= np.array(s[0:42]).reshape(6,7)
      c = s[:,act]

  

      if c[0] != 0 or self.value(a) != 0:
      # don't move in taken spots or in finished games
        
          rew = -10
    
      else:

      
        while c[j] != 0 :
                     j -=1   
                     
        c[j]=a[-1]                 
        rew += self.value(a)
      a[-1] = -a[-1]
      s[:,act]= c 
      b=a[-1]
      v=np.array(s).flatten()
      n=np.append(v,0)
      s=np.append(n,b)
    else:
        pass

    return rew, s

  

  def step(self, act):
    rew, self.state = self.dynamics(self.state, act)
    if rew != 0:
      self.done = True
    if np.all(np.array(self.state[0:42]) != 0):
      self.done = True
    return self.state, rew, self.done, None


# Play a quick round
env = ConnectFour()
print(env.reset())
print(env.step(4))
print(env.step(3))
print(env.step(3))
print(env.step(2))
print(env.step(2))
print(env.state[-1], env.value(env.state))


# In[3]:


# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *

def to_one_hot(x,n):
 ret = np.zeros([n])
 if x >= 0:
   ret[x] = 1.0
 return ret

def bstack(bb):
 ret = [[x] for x in bb[0]]
 for i in range(1, len(bb)):
   for j in range(len(bb[i])):
     ret[j].append(bb[i][j])
 return [np.array(x) for x in ret]

def reformat_batch(batch, a_dim, remove_policy=False):
 X,Y = [], []
 for o,a,outs in batch:
   x = [o] + [to_one_hot(x, a_dim) for x in a]
   y = []
   for ll in [list(x) for x in outs]:
     y += ll
   X.append(x)
   Y.append(y)
 X = bstack(X)
 Y = bstack(Y)
 if remove_policy:
   nY = [Y[0]]
   for i in range(7, len(Y), 6):
     nY.append(Y[i])
     nY.append(Y[i+1])
   Y = nY
 else:
   Y = [Y[0]] + Y[2:]
 return X,Y

class MuModel():
 LAYER_COUNT = 4
 LAYER_DIM = 128
 BN = False

 def __init__(self, o_dim, a_dim, s_dim=8, K=5, lr=0.001, with_policy=True):
   self.o_dim = o_dim
   self.a_dim = a_dim
   self.losses = []
   self.with_policy = with_policy

   # h: representation function
   # s_0 = h(o_1...o_t)
   x = o_0 = Input(o_dim)
   for i in range(self.LAYER_COUNT):
     x = Dense(self.LAYER_DIM, activation='elu')(x)
     if i != self.LAYER_COUNT-1 and self.BN:
       x = BatchNormalization()(x)
   s_0 = Dense(s_dim, name='s_0')(x)
   self.h = Model(o_0, s_0, name="h")

   # g: dynamics function (recurrent in state?) old_state+action -> state+reward
   # r_k, s_k = g(s_k-1, a_k)
   s_km1 = Input(s_dim)
   a_k = Input(self.a_dim)
   x = Concatenate()([s_km1, a_k])
   for i in range(self.LAYER_COUNT):
     x = Dense(self.LAYER_DIM, activation='elu')(x)
     if i != self.LAYER_COUNT-1 and self.BN:
       x = BatchNormalization()(x)
   s_k = Dense(s_dim, name='s_k')(x)
   r_k = Dense(1, name='r_k')(x)
   self.g = Model([s_km1, a_k], [r_k, s_k], name="g")

   # f: prediction function -- state -> policy+value
   # p_k, v_k = f(s_k)
   x = s_k = Input(s_dim)
   for i in range(self.LAYER_COUNT):
     x = Dense(self.LAYER_DIM, activation='elu')(x)
     if i != self.LAYER_COUNT-1 and self.BN:
       x = BatchNormalization()(x)
   v_k = Dense(1, name='v_k')(x)

   if self.with_policy:
     p_k = Dense(self.a_dim, name='p_k')(x)
     self.f = Model(s_k, [p_k, v_k], name="f")
   else:
     self.f = Model(s_k, v_k, name="f")

   # combine them all
   self.create_mu(K, lr)

 def ht(self, o_0):
   return self.h.predict(np.array(o_0)[None])[0]

 def gt(self, s_km1, a_k):
   r_k, s_k = self.g.predict([s_km1[None], to_one_hot(a_k, self.a_dim)[None]])
   return r_k[0][0], s_k[0]

 def ft(self, s_k):
   if self.with_policy:
     p_k, v_k = self.f.predict(s_k[None])
     return np.exp(p_k[0]), v_k[0][0]
   else:
     v_k = self.f.predict(s_k[None])
     return np.array([1/self.a_dim]*self.a_dim), v_k[0][0]

 def train_on_batch(self, batch):
   X,Y = reformat_batch(batch, self.a_dim, not self.with_policy)
   l = self.mu.train_on_batch(X,Y)
   self.losses.append(l)
   return l

 def create_mu(self, K, lr):
   self.K = K
   # represent
   o_0 = Input(self.o_dim, name="o_0")
   s_km1 = self.h(o_0)

   a_all, mu_all, loss_all = [], [], []

   def softmax_ce_logits(y_true, y_pred):
     return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

   # run f on the first state
   if self.with_policy:
     p_km1, v_km1 = self.f([s_km1])
     mu_all += [v_km1, p_km1]
     loss_all += ["mse", softmax_ce_logits]
   else:
     v_km1 = self.f([s_km1])
     mu_all += [v_km1]
     loss_all += ["mse"]

   for k in range(K):
     a_k = Input(self.a_dim, name="a_%d" % k)
     a_all.append(a_k)

     r_k, s_k = self.g([s_km1, a_k])

     # predict + store
     if self.with_policy:
       p_k, v_k = self.f([s_k])
       mu_all += [v_k, r_k, p_k]
       loss_all += ["mse", "mse", softmax_ce_logits]
     else:
       v_k = self.f([s_k])
       mu_all += [v_k, r_k]
       loss_all += ["mse", "mse"]
     
     # passback
     s_km1 = s_k

   mu = Model([o_0] + a_all, mu_all)
   mu.compile(Adam(lr), loss_all)
   self.mu = mu

import math
import random
import numpy as np

def softmax(x):
 e_x = np.exp(x - np.max(x))
 return e_x / e_x.sum()

class Node(object):
 def __init__(self, prior: float):
   self.visit_count = 0
   self.prior = prior
   self.value_sum = 0
   self.children = {}
   self.hidden_state = None
   self.reward = 0
   self.to_play = -1

 def expanded(self) -> bool:
   return len(self.children) > 0

 def value(self) -> float:
   if self.visit_count == 0:
     return 0
   return self.value_sum / self.visit_count

pb_c_base = 19652
pb_c_init = 1.25

discount = 0.95
root_dirichlet_alpha = 0.25
root_exploration_fraction = 0.25

class MinMaxStats(object):
 """A class that holds the min-max values of the tree."""

 def __init__(self):
   self.maximum = -float('inf')
   self.minimum = float('inf')

 def update(self, value: float):
   self.maximum = max(self.maximum, value)
   self.minimum = min(self.minimum, value)

 def normalize(self, value: float) -> float:
   if self.maximum > self.minimum:
     # We normalize only when we have set the maximum and minimum values.
     return (value - self.minimum) / (self.maximum - self.minimum)
   return value

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node, min_max_stats=None) -> float:
 pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
 pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

 prior_score = pb_c * child.prior
 if child.visit_count > 0:
   if min_max_stats is not None:
     value_score = child.reward + discount * min_max_stats.normalize(child.value())
   else:
     value_score = child.reward + discount * child.value()
 else:
   value_score = 0

 #print(prior_score, value_score)
 return prior_score + value_score

def select_child(node: Node, min_max_stats=None):
 out = [(ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items()]
 smax = max([x[0] for x in out])
 # this max is why it favors 1's over 0's
 _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
 return action, child

def mcts_search(m, observation, num_simulations=10, minimax=True):
 # init the root node
 root = Node(0)
 root.hidden_state = m.ht(observation)
 if minimax:
   root.to_play = observation[-1]
 policy, value = m.ft(root.hidden_state)

 # expand the children of the root node
 for i in range(policy.shape[0]):
   root.children[i] = Node(policy[i])
   root.children[i].to_play = -root.to_play

 # add exploration noise at the root
 actions = list(root.children.keys())
 noise = np.random.dirichlet([root_dirichlet_alpha] * len(actions))
 frac = root_exploration_fraction
 for a, n in zip(actions, noise):
   root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac

 # run_mcts
 min_max_stats = MinMaxStats()
 for _ in range(num_simulations):
   history = []
   node = root
   search_path = [node]

   # traverse down the tree according to the ucb_score 
   while node.expanded():
     action, node = select_child(node, min_max_stats)
     #action, node = select_child(node)
     history.append(action)
     search_path.append(node)

   # now we are at a leaf which is not "expanded", run the dynamics model
   parent = search_path[-2]
   node.reward, node.hidden_state = m.gt(parent.hidden_state, history[-1])

   # use the model to estimate the policy and value, use policy as prior
   policy, value = m.ft(node.hidden_state)
   #print(history, value)

   # create all the children of the newly expanded node
   for i in range(policy.shape[0]):
     node.children[i] = Node(prior=policy[i])
     node.children[i].to_play = -node.to_play

   # update the state with "backpropagate"
   for bnode in reversed(search_path):
     if minimax:
       bnode.value_sum += value if root.to_play == bnode.to_play else -value
     else:
       bnode.value_sum += value
     bnode.visit_count += 1
     min_max_stats.update(node.value())
     value = bnode.reward + discount * value

 # output the final policy
 visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
 visit_counts = [x[1] for x in sorted(visit_counts)]
 av = np.array(visit_counts).astype(np.float64)
 policy = softmax(av)
 return policy, root

def print_tree(x, hist=[]):
 if x.visit_count != 0:
   print("%3d %4d %-16s %8.4f %4d" % (x.to_play, x.visit_count, str(hist), x.value(), x.reward))
 for i,c in x.children.items():
   print_tree(c, hist+[i])

def get_action_space(K, n):
 def to_one_hot(x,n):
   ret = np.zeros([n])
   ret[x] = 1.0
   return ret
 import itertools
 aopts = list(itertools.product(list(range(n)), repeat=K))
 aoptss = np.array([[to_one_hot(x, n) for x in aa] for aa in aopts])
 aoptss = aoptss.swapaxes(0,1)
 aoptss = [aoptss[x] for x in range(K)]
 return aopts,aoptss


import numpy as np
import random

class Game():
 def __init__(self, env, discount=0.95):
   self.env = env
   self.observations = []
   self.history = []
   self.rewards = []
   self.policies = []
   self.discount = discount
   self.done = False
   self.observation = env.reset()
   self.total_reward = 0

 def terminal(self):
   return self.done

 def apply(self, a_1, p=None):
   self.observations.append(np.copy(self.observation))
   self.observation, r_1, done, _ = self.env.step(a_1)

   self.history.append(a_1)
   self.rewards.append(r_1)
   self.total_reward += r_1
   self.policies.append(p)

   self.done = done

 def act_with_policy(self, policy):
   act = np.random.choice(list(range(len(policy))), p=policy)
   self.apply(act, policy)

 def make_image(self, i):
   return self.observations[i]

 def make_target(self, state_index, num_unroll_steps):
   targets = []
   for current_index in range(state_index, state_index + num_unroll_steps + 1):
     value = 0
     for i, reward in enumerate(self.rewards[current_index:]):
       value += reward * self.discount**i
       
     if current_index > 0 and current_index <= len(self.rewards):
       last_reward = self.rewards[current_index - 1]
     else:
       last_reward = 0

     if current_index < len(self.policies):
       targets.append((value, last_reward, self.policies[current_index]))
     else:
       # no policy, what does cross entropy do? hopefully not learn
       targets.append((0, last_reward, np.array([0]*len(self.policies[0]))))
   return targets 

class ReplayBuffer():
 def __init__(self, window_size, batch_size, num_unroll_steps):
   self.window_size = window_size
   self.batch_size = batch_size
   self.num_unroll_steps = num_unroll_steps
   self.buffer = []

 def save_game(self, game):
   if len(self.buffer) > self.window_size:
     self.buffer.pop(0)
   self.buffer.append(game)

 def sample_batch(self, bs=None):
   games = [self.sample_game() for _ in range(self.batch_size if bs is None else bs)]
   game_pos = [(g, self.sample_position(g)) for g in games]
   def xtend(g,x,s):
     # pick the last (fake) action
     while len(x) < s:
       #x.append(random.randint(0, len(g.policies[0])-1))
       #x.append(len(g.policies[0])-1)
       x.append(-1)
     return x
   return [(g.make_image(i), xtend(g,g.history[i:i + self.num_unroll_steps], self.num_unroll_steps),
            g.make_target(i, self.num_unroll_steps))
            for (g, i) in game_pos]

 def sample_game(self):
   
   #without priority sampling
   return random.choice(self.buffer)
  

 def sample_position(self, game):
   # have to do -num_unroll_steps to allow enough actions
   return random.randint(0, len(game.history)-1) 


# In[4]:


class MockModel():
  def ht(self,s):
    return s
  def gt(self, s, a):
    #print(s, a)
    return env.dynamics(s,a)
  def ft(self,s):
    #print(s, env.value(s))
    return np.array([1/7]*7), env.value(s)

# unit tests for the MCTS!
mm = MockModel()


# In[5]:



gg = ConnectFour()
done = False
while not done:
  policy, node = mcts_search(mm, gg.state, 2000)
  print(policy)
  act = np.random.choice(list(range(len(policy))), p=policy)
  print(act)
  _, _, done, _ = gg.step(act)
  gg.render()


# In[6]:



m = MuModel(env.observation_space.shape, env.action_space.n, s_dim=64, K=5, lr=0.001)
print(env.observation_space.shape, env.action_space.n)

replay_buffer = ReplayBuffer(200, 16, m.K)
rews = []


# In[7]:


def play_game(env, m):
  import random
  game = Game(env, discount=0.99)
  while not game.terminal():
    policy, _ = mcts_search(m, game.observation)
    game.act_with_policy(policy)
  return game


# In[8]:



import collections

for j in range(20):
  game = play_game(env, m)
  replay_buffer.save_game(game)
  for i in range(20):
    m.train_on_batch(replay_buffer.sample_batch())
  rew = sum(game.rewards)
  rews.append(rew)
  print(len(game.history), rew, game.history, m.losses[-1][0])


# In[9]:



plot(rews)
figure()
plot([x[-3] for x in m.losses])


# In[10]:



# show starting policy

obs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
a=ConnectFour(obs).render()
policy, value = m.ft(m.ht(obs))
np.array(policy[0:42]).reshape(6,7)


# In[16]:


obs = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, -1, 1, -1, 1, -1,1,0, 0,1]
ConnectFour(obs).render()
policy, value = m.ft(m.ht(obs))


rew = [m.gt(m.ht(obs), i)[0] for i in range(42)]

np.array(policy[0:42]).reshape(6,7)
np.array(rew[0:42]).reshape(6,7)


# In[ ]:




