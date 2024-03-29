# -*- coding: utf-8 -*-
"""RLFinalESARSAGridWorld.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ij8Q1I8NGdm8JHrskVoVT1s86Qxtihhd
"""

import numpy as np

def go_up(current_state):
  return (current_state[0]-1, current_state[1])

def go_down(current_state):
  return (current_state[0]+1, current_state[1])

def go_left(current_state):
  return (current_state[0], current_state[1]-1)

def go_right(current_state):
  return (current_state[0], current_state[1]+1)

def get_reward(next_state, gold):
  if next_state == (0, 2):
    return gold
  if next_state == (4, 2):
    return -10
  if next_state == (4, 4):
    return 10
  return 0

def is_valid_state(current_state):
  return current_state[0] > -1 and current_state[0] < 5 and current_state[1] > -1 and current_state[1] < 5 and current_state != (2, 2) and current_state != (3, 2)

def add_to_dic(dic, state, current_state, p):
  if is_valid_state(state):
    if state in dic.keys():
      dic[state] += p
    else:
      dic[state] = p
  else:
    if current_state in dic.keys():
      dic[current_state] += p
    else:
      dic[current_state] = p

def get_transition_probs(dic, main_state, left_state, right_state, current_state):
    add_to_dic(dic, main_state, current_state, 0.8)
    add_to_dic(dic, left_state, current_state, 0.05)
    add_to_dic(dic, right_state, current_state, 0.05)
    if current_state in dic.keys():
        dic[current_state] += 0.10
    else:
      dic[current_state] = 0.10

def get_next_state_probs(current_state, action):
  dic = {}
  if action == 'AU':
    main_state = go_up(current_state)
    left_state = go_left(current_state)
    right_state = go_right(current_state)
    get_transition_probs(dic, main_state, left_state, right_state, current_state)

  if action == 'AD':
    main_state = go_down(current_state)
    left_state = go_right(current_state)
    right_state = go_left(current_state)
    get_transition_probs(dic, main_state, left_state, right_state, current_state)

  if action == 'AL':
    main_state = go_left(current_state)
    left_state = go_down(current_state)
    right_state = go_up(current_state)
    get_transition_probs(dic, main_state, left_state, right_state, current_state)

  if action == 'AR':
    main_state = go_right(current_state)
    left_state = go_up(current_state)
    right_state = go_down(current_state)
    get_transition_probs(dic, main_state, left_state, right_state, current_state)

  return dic

def arrow_representation(state_actions):
  dic = {'AU':'\u2191', 'AD':'\u2193', 'AL':'\u2190', 'AR':'\u2192', 'G':'G'}
  for i in range(len(state_actions)):
    for j in range(len(state_actions)):
      state_actions[i][j] = dic.get(state_actions[i][j], state_actions[i][j])
  return state_actions

def rounded_representation(state_values):
  for i in range(len(state_values)):
    for j in range(len(state_values)):
      state_values[i][j] = round(state_values[i][j], 4)
  return state_values

def get_next_state(current_state, current_action):
  next_state_probs = get_next_state_probs(current_state, current_action)
  next_state_index = np.random.choice([i for i in range(len(next_state_probs.values()))], p=list(next_state_probs.values()))
  next_state = list(next_state_probs.keys())[next_state_index]
  return next_state

def algorithm_three_sin(weights, current_state):
  phi = get_s_vector(current_state)
  q_values = np.dot(weights, phi)
  if q_values[0] == max(q_values):
    p = [1 - epsilon + (epsilon/4), epsilon/4, epsilon/4, epsilon/4]
  elif q_values[1] == max(q_values):
    p = [epsilon/4, 1 - epsilon + (epsilon/4), epsilon/4, epsilon/4]
  elif q_values[2] == max(q_values):
    p = [epsilon/4, epsilon/4, 1 - epsilon + (epsilon/4), epsilon/4]
  else:
    p = [epsilon/4, epsilon/4, epsilon/4, 1 - epsilon + (epsilon/4)]
  action_index = np.random.choice([0, 1, 2, 3], p=p)
  return get_action_from_index(action_index)

def get_action_from_index(action_index):
  if action_index == 0:
    return 'AU'
  elif action_index == 1:
    return 'AD'
  elif action_index == 2:
    return 'AL'
  else:
    return 'AR'

def get_max_action_SARSA(weights, current_state):
  phi = get_s_vector(current_state)
  q_values = np.dot(weights, phi)
  if q_values[0] == max(q_values):
    return 'AU'
  elif q_values[1] == max(q_values):
    return 'AD'
  elif q_values[2] == max(q_values):
    return 'AL'
  else:
    return 'AR'

def n_step_return(n, tau, T):
  G = 0
  for i in range(tau+1, min(tau+n, T)+1):
    G += (gamma**(i-tau-1))*rewards[i]
  return G

def q_fnc(state, action):
  state_vector = get_s_vector(state)
  if action == 'AU':
    action_vector = weights[0].copy()
  elif action == 'AD':
    action_vector = weights[1].copy()
  elif action == 'AL':
    action_vector = weights[2].copy()
  else:
    action_vector = weights[3].copy()
  return np.dot(state_vector, action_vector)

def calculate_policy(weights):
  l = np.random.choice(['AU'], size=(5, 5), p=[1])
  for i in range(5):
    for j in range(5):
      if (i, j) == (4, 4):
        l[i, j] = 'G'
        continue
      if (i, j) in [(2, 2), (3, 2)]:
        l[i, j] = ' '
        continue
      phi = get_s_vector((i, j))
      q_values = np.dot(weights, phi)
      max_ind = np.where(q_values == max(q_values))[0][0]
      l[i, j] = get_action_from_index(max_ind)
  return l

def run_algo(weights):
  c = 0
  current_state = (0 ,0)
  while current_state not in [(4, 4)]:
    current_action = get_max_action_SARSA(weights, current_state)
    next_state = get_next_state(current_state, current_action)
    current_state = next_state
    c += 1
    if c > 50:
      return c
  return c

def get_s_vector(state):
  ind = (state[0] * 5) + state[1]
  state_vec = np.zeros(25)
  state_vec[ind] = 1
  return state_vec

######## Episodic n-step
import itertools
import math
alpha = 0.005
epsilon = 0.3
gamma = 0.9
gold = 0
n = 1

weights = np.random.rand(4, 25)
counts = []
for k in range(20000):
  count = 0
  states = {}
  actions = {}
  rewards = {}
  current_state = (2, 2)
  while (not is_valid_state(current_state) and current_state not in [(4, 4)]):
    current_state = (np.random.randint(0, 5), np.random.randint(0, 5))
  states[0] = current_state
  current_action = algorithm_three_sin(weights, current_state)
  actions[0] = current_action
  T = float('inf')
  for t in itertools.count():
    if t < T:
      A_t = actions[t]
      next_state = get_next_state(current_state, current_action)
      states[t+1] = next_state
      reward = get_reward(next_state, gold)
      rewards[t+1] = reward
      if reward == 10:
        T = t+1
      else:
        next_action = algorithm_three_sin(weights, next_state)
        actions[t+1] = next_action
    tau = t - n + 1
    if tau >= 0:
      G = n_step_return(n, tau, T)
      if tau + n < T:
        G += (gamma ** n) * q_fnc(states[tau+n], actions[tau+n])
      if current_action == 'AU':
        weights[0] += alpha * (G - q_fnc(states[tau], actions[tau])) * np.array(get_s_vector(states[tau]))
      if current_action == 'AD':
        weights[1] += alpha * (G - q_fnc(states[tau], actions[tau])) * np.array(get_s_vector(states[tau]))
      if current_action == 'AL':
        weights[2] += alpha * (G - q_fnc(states[tau], actions[tau])) * np.array(get_s_vector(states[tau]))
      else:
        weights[3] += alpha * (G - q_fnc(states[tau], actions[tau])) * np.array(get_s_vector(states[tau]))
    if tau == T-1:
      counts.append(run_algo(weights))
      break
    current_state = next_state
    current_action = next_action
    count += 1

import matplotlib.pyplot as plt



# Adding labels to the axes
plt.xlabel('Episodes')
plt.ylabel('Steps to complete GridWorld')

# Adding a title to the graph
plt.title('Policy Performance (One Hot Encoding, alpha=0.005, epsilon=0.3, n=1)')

plt.plot(counts)
# Displaying the graph
plt.show()
print(arrow_representation(calculate_policy(weights)))