# -*- coding: utf-8 -*-
"""RLHW4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xbrkTtsOGm-zZVHufwl4ZWAW_wWyonO0
"""

import numpy as np
import matplotlib.pyplot as plt

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

def get_max_action(current_state, prev_values, gamma, gold, terminal_states):
  max_action = ''
  max_action_value = float('-inf')
  if current_state in terminal_states:
    return 'G', 0
  for action in ['AU', 'AD', 'AL', 'AR']:
    a = get_next_state_probs(current_state, action)
    total = 0
    for key, value in a.items():
      total += (value*get_reward(key, gold)) + (value*gamma*prev_values[key])
    if total > max_action_value:
      max_action = action
      max_action_value = total
  return max_action, max_action_value

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
  dic = get_next_state_probs(current_state, current_action)
  ind = np.random.choice(list(range(len(dic.values()))), p=list(dic.values()))
  next_state = list(dic.keys())[ind]
  return next_state

def get_q_state_index(state):
  return state[0] * 5 + state[1]

def get_q_action_index(action):
  if action == 'AU':
    return 0
  elif action == 'AD':
    return 1
  elif action == 'AL':
    return 2
  else:
    return 3

def get_action_from_index(action_index):
    if action_index == 0:
      return 'AU'
    if action_index == 1:
      return 'AD'
    if action_index == 2:
      return 'AL'
    else:
      return 'AR'

def get_epsilon_greedy_action(q_values, state, epsilon):
  max_ind = np.where(q_values[get_q_state_index(state)] == max(q_values[get_q_state_index(state)]))[0][0]
  choices = list(range(4))
  choices.remove(max_ind)
  choices.insert(0, max_ind)
  p = [1 - epsilon + (epsilon/4), epsilon/4, epsilon/4, epsilon/4]
  action_index = np.random.choice(choices, p=p)
  return get_action_from_index(action_index)

def get_policy(q_values):
  lst = []
  empty_2d_array = np.random.choice(['AU', 'AD', 'AL', 'AR'], size=(5, 5), p=[0.25, 0.25, 0.25, 0.25])
  for i in range(len(q_values)):
    lst.append(np.where(q_values[i] == max(q_values[i]))[0][0])
  lst.reverse()

  for i in range(5):
    for j in range(5):
      if not is_valid_state((i, j)):
        empty_2d_array[(i, j)] = ' '
        lst.pop()
        continue
      if (i, j) in terminal_states:
        empty_2d_array[(i, j)] = 'G'
        lst.pop()
        continue
      empty_2d_array[(i, j)] = get_action_from_index(lst.pop())

  print(arrow_representation(empty_2d_array))

def get_SARSA_value_fnc(q_values):
  vals = []
  for k in range(len(q_values)):
    m = max(q_values[k])
    s = 0
    for i in range(len(q_values[k])):
      if q_values[k][i] == m:
        s += (1 - epsilon + (epsilon/4)) * q_values[k][i]
      else:
        s += (epsilon/4 * q_values[k][i])
    vals.append(s)
  vals.reverse()
  val_fnc = np.zeros((5, 5))
  for i in range(5):
    for j in range(5):
      val_fnc[i, j] = vals.pop()
  return val_fnc

def get_QLearning_value_fnc(q_values):
  vals = []
  for k in range(len(q_values)):
    vals.append(max(q_values[k]))
  vals.reverse()
  val_fnc = np.zeros((5, 5))
  for i in range(5):
    for j in range(5):
      val_fnc[i, j] = vals.pop()
  return val_fnc

def get_mean_square_error(value_fns, optimal_value_fn):
  errors = []
  for fn in value_fns:
    errors.append(np.mean((fn - optimal_value_fn) ** 2))
  return errors

#############################################################################################
#################################### Q1 #####################################################
#############################################################################################

state_values = np.zeros((5, 5))
next_state_values = np.zeros((5, 5))
state_actions = np.random.choice(['AU', 'AD', 'AL', 'AR'], size=(5, 5), p=[0.25, 0.25, 0.25, 0.25])
state_actions[(2, 2)] = ' '
state_actions[(3, 2)] = ' '
gamma = 0.9
gold = 0
terminal_states = [(4, 4)]

counter = 0
while True:
  delta = 0
  counter += 1
  for i in range(len(state_values)):
    for j in range(len(state_values)):
      if is_valid_state((i, j)):
        temp = state_values[i][j]
        state_actions[i][j], next_state_values[i][j] = get_max_action((i, j), state_values, gamma, gold, terminal_states)
        delta = max(delta, abs(temp - next_state_values[i][j]))
  if delta < 0.0001:
    break
  state_values = next_state_values.copy()
vi_values = rounded_representation(state_values)
print(vi_values)
#print(arrow_representation(state_actions))
print("Iterations ->"+str(counter))

#############################################################################################
#################################### TD #####################################################
#############################################################################################

gamma = 0.9
gold = 0
alpha = 0.25
terminal_states = [(4, 4)]

values_estimator = []
episodes = []
for _ in range(50):
  state_values = np.zeros((5, 5))
  next_state_values = np.zeros((5, 5))
  episode_counter = 0
  while True:
    current_state = (2, 2)
    while (not is_valid_state(current_state) and current_state not in terminal_states):
      current_state = (np.random.randint(0, 5), np.random.randint(0, 5))
    while True:
      if current_state in terminal_states:
        break
      action = state_actions[current_state]
      next_state = get_next_state(current_state, action)
      reward = get_reward(next_state, gold)
      next_state_values[current_state] = next_state_values[current_state] + (alpha * (reward + (gamma * next_state_values[next_state]) - next_state_values[current_state]))
      current_state = next_state
    episode_counter += 1
    max_norm_distance = np.max(np.abs(state_values - next_state_values))
    if max_norm_distance < 0.001 and max_norm_distance != 0:
      values_estimator.append(next_state_values)
      episodes.append(episode_counter)
      break
    state_values = next_state_values.copy()
avg_value_function = np.mean(np.array(values_estimator), axis=0)
td_values = rounded_representation(avg_value_function)
print('TD Value Function ->')
print(td_values)
print('Max Norm Distance ->')
print(np.max(np.abs(vi_values - td_values)))
print('Avg Episodes ->')
print(np.mean(episodes))
print('Standard Deviation Episodes ->')
print(np.std(episodes))

#############################################################################################
#################################### SARSA ##################################################
#############################################################################################

alpha = 0.1
epsilon = 0.5
gamma = 0.9
gold = 0
terminal_states = [(4, 4)]
episode_counts = []
every_episode_q_values_outer = []
for _ in range(20):
  count = 0
  q_values = np.zeros((25, 4))
  next_q_values = np.zeros((25, 4))
  episode_actions = []
  every_episode_q_values_inner = []
  action_counter = 0
  while True:
    count += 1
    current_state = (2, 2)
    while (not is_valid_state(current_state) and current_state not in terminal_states):
      current_state = (np.random.randint(0, 5), np.random.randint(0, 5))
    current_action = get_epsilon_greedy_action(next_q_values, current_state, epsilon)
    action_counter += 1
    while True:
      if current_state in terminal_states:
        episode_actions.append(action_counter)
        every_episode_q_values_inner.append(next_q_values.copy())
        break
      next_state = get_next_state(current_state, current_action)
      reward = get_reward(next_state, gold)
      next_action = get_epsilon_greedy_action(next_q_values, next_state, epsilon)
      action_counter += 1
      next_q_values[get_q_state_index(current_state), get_q_action_index(current_action)] = next_q_values[get_q_state_index(current_state), get_q_action_index(current_action)] + (alpha * (reward + (gamma * next_q_values[get_q_state_index(next_state), get_q_action_index(next_action)]) - next_q_values[get_q_state_index(current_state), get_q_action_index(current_action)]))
      current_state = next_state
      current_action = next_action
    if count == 4000:
      episode_counts.append(episode_actions.copy())
      every_episode_q_values_outer.append(every_episode_q_values_inner.copy())
      break
    q_values = next_q_values.copy()


#############################################################################################
#################################### 2a #####################################################
#############################################################################################

avg_episode_counts = np.mean(episode_counts, axis=0)
x = avg_episode_counts
y = range(len(avg_episode_counts))
plt.plot(x, y)
plt.xlabel('Timesteps/Actions')
plt.ylabel('Episodes')
plt.title('Episodes vs Timesteps Graph')
plt.show()

#############################################################################################
#################################### 2b #####################################################
#############################################################################################

every_episode_q_values = []
for i in range(4000):
  s = []
  for j in range(20):
    s.append(every_episode_q_values_outer[j][i].copy())
  every_episode_q_values.append(np.mean(s, axis=0))

every_episode_value_fn = []
for i in range(len(every_episode_q_values)):
  every_episode_value_fn.append(get_SARSA_value_fnc(every_episode_q_values[i]))

mse = get_mean_square_error(every_episode_value_fn, vi_values)

x = range(len(mse))
y = mse
plt.plot(x, y)
plt.ylabel('Average MSE per episode')
plt.xlabel('Episodes')
plt.title('Episodes vs MSE Graph')
plt.show()

#############################################################################################
#################################### 2c #####################################################
#############################################################################################

print('Learnt Policy ->')
print(get_policy(every_episode_q_values[3999]))

mse[3999]

#############################################################################################
#################################### QLearning ##############################################
#############################################################################################

alpha = 0.1
epsilon = 0.5
gamma = 0.9
gold = 0
terminal_states = [(4, 4)]
episode_counts = []
every_episode_q_values_outer = []
for _ in range(20):
  count = 0
  q_values = np.zeros((25, 4))
  next_q_values = np.zeros((25, 4))
  episode_actions = []
  every_episode_q_values_inner = []
  action_counter = 0
  while True:
    count += 1
    current_state = (2, 2)
    while (not is_valid_state(current_state) and current_state not in terminal_states):
      current_state = (np.random.randint(0, 5), np.random.randint(0, 5))
    while True:
      if current_state in terminal_states:
        episode_actions.append(action_counter)
        every_episode_q_values_inner.append(next_q_values.copy())
        break
      current_action = get_epsilon_greedy_action(next_q_values, current_state, epsilon)
      action_counter += 1
      next_state = get_next_state(current_state, current_action)
      reward = get_reward(next_state, gold)
      next_q_values[get_q_state_index(current_state), get_q_action_index(current_action)] = next_q_values[get_q_state_index(current_state), get_q_action_index(current_action)] + (alpha * (reward + (gamma * max(next_q_values[get_q_state_index(next_state)])) - next_q_values[get_q_state_index(current_state), get_q_action_index(current_action)]))
      current_state = next_state

    if count == 4000:
      episode_counts.append(episode_actions.copy())
      every_episode_q_values_outer.append(every_episode_q_values_inner.copy())
      break
    q_values = next_q_values.copy()


#############################################################################################
#################################### 3a #####################################################
#############################################################################################

avg_episode_counts = np.mean(episode_counts, axis=0)
x = avg_episode_counts
y = range(len(avg_episode_counts))
plt.plot(x, y)
plt.xlabel('Timesteps/Actions')
plt.ylabel('Episodes')
plt.title('Episodes vs Timesteps Graph')
plt.show()

############################################################################################
################################### 3b #####################################################
############################################################################################

every_episode_q_values = []
for i in range(4000):
  s = []
  for j in range(20):
    s.append(every_episode_q_values_outer[j][i].copy())
  every_episode_q_values.append(np.mean(s, axis=0))

every_episode_value_fn = []
for i in range(len(every_episode_q_values)):
  every_episode_value_fn.append(get_QLearning_value_fnc(every_episode_q_values[i]))

mse = get_mean_square_error(every_episode_value_fn, vi_values)

x = range(len(mse))
y = mse
plt.plot(x, y)
plt.ylabel('Average MSE per episode')
plt.xlabel('Episodes')
plt.title('Episodes vs MSE Graph')
plt.show()

#############################################################################################
#################################### 3c #####################################################
#############################################################################################

print('Learnt Policy ->')
print(get_policy(every_episode_q_values[3999]))