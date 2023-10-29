import math
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

def get_next_state(F, current_state):
  g = 9.8
  mc = 1.0
  mp = 0.1
  mt = 1.1
  l = 0.5
  T = 0.02
  cart_pos, cart_vel, pole_angle, pole_vel = current_state
  b = (F + (mp*l*(pole_vel**2)*math.sin(pole_angle)))/mt
  c = ((g*math.sin(pole_angle)) - (math.cos(pole_angle)*b))/(l*((4/3) - ((mp*(math.cos(pole_angle)**2))/mt)))
  d = b - ((mp*l*c*math.cos(pole_angle))/mt)
  next_cart_pos = cart_pos + (T*cart_vel)
  next_cart_vel = cart_vel + (T*d)
  next_pole_angle = pole_angle + (T*pole_vel)
  next_pole_vel = pole_vel + (T*c)
  next_state = (next_cart_pos, next_cart_vel, next_pole_angle, next_pole_vel)
  return next_state

def terminate(current_state):
  return (current_state[2] < -math.pi/15) or (current_state[2] > math.pi/15) or (current_state[0] < -2.4) or (current_state[0] > 2.4)

def algorithm_one(current_pol_vec, nPerturbations, sigma, N, alpha):
  policies = []
  n = len(current_pol_vec)
  O = np.zeros(n)
  I = np.eye(n)
  for _ in range(100): #Change episodes
    epsilons = []
    estimates = []
    for _ in range(nPerturbations):
      epsilon = np.random.multivariate_normal(O, I, size=1)[0]
      epsilons.append(epsilon)
      temp_pol = current_pol_vec + (sigma*epsilon)
      estimate = algorithm_two(temp_pol, N)
      estimates.append(estimate)
    prod = 0
    for i in range(len(estimates)):
      prod += epsilons[i] * estimates[i]
    next_pol_vec = current_pol_vec + (alpha*(1/(sigma*nPerturbations))*prod)
    policies.append(run_policy(next_pol_vec))
    current_pol_vec = next_pol_vec
  return policies

def algorithm_two(pol_vec, N):
  returns = []
  for i in range(N):
    pol_return = run_policy(pol_vec)
    returns.append(pol_return)
  return sum(returns)/N

def run_policy(pol_vec):
  force = {'L':-10, 'R':10}
  M = int((len(pol_vec) - 1)/4)
  current_state = (0, 0, 0, 0)
  reward = 0
  while True:
    if terminate(current_state):
      break
    if reward >= 500:
      break
    reward += 1
    action = algorithm_three_sin(pol_vec, current_state, M)
    next_state = get_next_state(force[action], current_state)
    current_state = next_state
  return reward

# def algorithm_three_cos(pol_vec, current_state, M):
#   normalized_state = normalize_cos(current_state)
#   phi = [1]
#   for i in range(1, M+1, 1):
#     phi.append(math.cos(i*math.pi*normalized_state[0]))
#   for i in range(1, M+1, 1):
#     phi.append(math.cos(i*math.pi*normalized_state[1]))
#   for i in range(1, M+1, 1):
#     phi.append(math.cos(i*math.pi*normalized_state[2]))
#   for i in range(1, M+1, 1):
#     phi.append(math.cos(i*math.pi*normalized_state[3]))
#   threshold = np.dot(np.array(phi), pol_vec)
#   if threshold <= 0:
#     return 'L'
#   else:
#     return 'R'

def algorithm_three_sin(pol_vec, current_state, M):
  normalized_state = normalize_sin(current_state)
  phi = [1]
  for i in range(1, M+1, 1):
    phi.append(math.sin(i*math.pi*normalized_state[0]))
  for i in range(1, M+1, 1):
    phi.append(math.sin(i*math.pi*normalized_state[1]))
  for i in range(1, M+1, 1):
    phi.append(math.sin(i*math.pi*normalized_state[2]))
  for i in range(1, M+1, 1):
    phi.append(math.sin(i*math.pi*normalized_state[3]))
  threshold = np.dot(np.array(phi), pol_vec)
  if threshold <= 0:
    return 'L'
  else:
    return 'R'

# def normalize_cos(state):
#   if (state[0] > 2.4) or (state[0] < -2.4):
#     raise Exception("cart_pos is "+str(state[0]))
#   if (state[1] > 4.6) or (state[1] < -4.6):
#     raise Exception("cart_vel is "+str(state[1]))
#   if (state[2] > math.pi/15) or (state[2] < -math.pi/15):
#     raise Exception("pole_angle is "+str(state[2]))
#   if (state[3] > 3.65) or (state[3] < -3.65):
#     raise Exception("pole_ang_vel is "+str(state[3]))
#   norm_cart_pos = (state[0] + 2.4)/4.8
#   norm_cart_vel = (state[1] + 4.6)/9.2
#   norm_pole_angle = (state[2] + (math.pi/15))/(2*math.pi/15)
#   norm_pole_ang_vel = (state[3] + 3.65)/7.3
#   normalized_state = (norm_cart_pos, norm_cart_vel, norm_pole_angle, norm_pole_ang_vel)
#   return normalized_state

def normalize_sin(state):
  if (state[0] > 2.4) or (state[0] < -2.4):
    raise Exception("cart_pos is "+str(state[0]))
  if (state[1] > 4.9) or (state[1] < -4.9):
    raise Exception("cart_vel is "+str(state[1]))
  if (state[2] > math.pi/15) or (state[2] < -math.pi/15):
    raise Exception("pole_angle is "+str(state[2]))
  if (state[3] > 3.65) or (state[3] < -3.65):
    raise Exception("pole_ang_vel is "+str(state[3]))
  norm_cart_pos = (((state[0] + 2.4)/4.8) * 2) - 1
  norm_cart_vel = (((state[1] + 4.9)/9.8) * 2) - 1
  norm_pole_angle = (((state[2] + (math.pi/15))/(2*math.pi/15)) * 2) - 1
  norm_pole_ang_vel = (((state[3] + 3.65)/7.3) * 2) - 1
  normalized_state = (norm_cart_pos, norm_cart_vel, norm_pole_angle, norm_pole_ang_vel)
  return normalized_state

def plot_average_return_graph(iterations, y, nPerturbations, M, sigma, N, alpha):
  x = range(1, len(y)+1, 1)
  plt.plot(x, y)
  plt.xlabel('No of episodes')
  plt.ylabel('Average Return')
  plt.title('{} iterations ES(nPerturbations = {}, M = {}, sigma={}, N={} alpha={})'.format(iterations, nPerturbations, M, sigma, N, alpha))
  plt.show()

def run_trials_es(iterations, nPerturbations, M, sigma, N, alpha):
  lst = []
  for _ in range(iterations):
    current_pol_vec = np.random.rand((M*4)+1)
    lst.append(np.array(algorithm_one(current_pol_vec, nPerturbations, sigma, N, alpha)))
  c = reduce(lambda x, y: x+y, lst)/iterations
  plot_average_return_graph(iterations, c, nPerturbations, M, sigma, N, alpha)

############### Q1, Trial 1 #######################

M = 20
nPerturbations = 50
sigma = 0.1
N = 1
alpha = 0.1
iterations = 5
run_trials_es(iterations, nPerturbations, M, sigma, N, alpha)

############### Q1, Trial 2 #######################

M = 50
nPerturbations = 80
sigma = 0.1
N = 1
alpha = 0.1
iterations = 5
run_trials_es(iterations, nPerturbations, M, sigma, N, alpha)

############### Q1, Trial 3 #######################

M = 5
nPerturbations = 80
sigma = 0.1
N = 1
alpha = 0.001
iterations = 5
run_trials_es(iterations, nPerturbations, M, sigma, N, alpha)

############### Q1, Trial 4 #######################

M = 5
nPerturbations = 80
sigma = 0.9
N = 1
alpha = 0.01
iterations = 5
run_trials_es(iterations, nPerturbations, M, sigma, N, alpha)

############### Q1, Trial 5 #######################

M = 5
nPerturbations = 80
sigma = 0.5
N = 1
alpha = 0.001
iterations = 5
run_trials_es(iterations, nPerturbations, M, sigma, N, alpha)

################ Q2 #######################

M = 5
nPerturbations = 80
sigma = 0.5
N = 1
alpha = 0.001

lst = []
for _ in range(20):
  current_pol_vec = np.random.rand((M*4)+1)
  lst.append(np.array(algorithm_one(current_pol_vec, nPerturbations, sigma, N, alpha)))
c = reduce(lambda x, y: x+y, lst)/20
std_deviation = np.std(lst, axis=0)

print(c)
print(std_deviation)
x = range(1, len(c)+1, 1)
plt.plot(x, c, label='Average Return', zorder=2)
plt.errorbar(x, c, yerr=std_deviation, label='Standard Deviation', fmt='o', ecolor='red', capsize=1, zorder=1)
plt.xlabel('No of episodes')
plt.ylabel('Average Return')
plt.title('20 Iterations ES w/ Standard Deviation')
plt.legend()
plt.show()
