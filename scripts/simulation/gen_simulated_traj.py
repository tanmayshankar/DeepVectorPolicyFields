"""Script to generate simulated trajectories for DVPF."""
import argparse
import copy
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D

# Hyperparameters
N = 5000
mu_error = 0.0
# sigma_error = 0.005
sigma_error = 0.1
# ALPHA = 0.5
ALPHA = 0.5
BETA = 0.001
# SIGMA = 0.005
SIGMA = 0.1
# GAMMA = 0.1
GAMMA = 0.3

def gaussian_error():
  """
  error Model: Epsilon ~ Normal(mu_error, sigma_error)
  """
  eps = np.random.normal(mu_error, sigma_error, N)
  return eps

def gaussian_sine_error():
  """
  error model:  Epsilon ~ Normal(BETA * sin(ALPHA*t),SIGMA)
  """
  t = np.array(range(0,N))
  eps_mu = BETA * np.sin(ALPHA * t)
  eps_sigma = np.tile(SIGMA, N)
  eps = np.random.normal(eps_mu, eps_sigma, N)
  return eps

def gaussian_sine_error_vel(v):
  """
  error model:  Epsilon ~ Normal(BETA * sin(ALPHA*t),GAMMA * v)
  """
  t = np.array(range(0,N))
  eps_mu = GAMMA * BETA * np.sin(ALPHA * t)
  eps_sigma = GAMMA * v
  eps = np.random.normal(eps_mu, eps_sigma, N)
  return eps

def add_noise_1(traj_commanded, traj_actual):
  """
  Add noise to state x_t directly
  """
  K = len(traj_actual)
  velocity_commanded = np.zeros([K,N-1])
  velocity_actual = np.zeros([K,N-1])

  for i in range(K):
    traj_actual[i] += gaussian_error()
    for j in range(1,N):
      velocity_commanded[i][j-1] = traj_commanded[i][j] - traj_commanded[i][j-1]
      velocity_actual[i][j-1] = traj_actual[i][j] - traj_actual[i][j-1]
  return velocity_commanded, velocity_actual

def add_noise_2(traj_commanded, traj_actual):
  """
  Add (Gaussian sine) noise to velocity velocity v_t.
  """
  K = len(traj_actual)
  velocity_commanded = np.zeros([K,N-1])
  velocity_actual = np.zeros([K,N-1])
 
  for i in range(K):
    eps = gaussian_sine_error()
    for j in range(1,N):
      # Recurrance (prime below means actual traj with error):
      # x_{t+1}^prime = x_{t}^prime + x_{t+1} - x_{t} + eps
      traj_actual[i][j] = traj_actual[i][j-1] + traj_commanded[i][j] - traj_commanded[i][j-1] + eps[j]
      velocity_commanded[i][j-1] = traj_commanded[i][j] - traj_commanded[i][j-1]
      velocity_actual[i][j-1] = traj_actual[i][j] - traj_actual[i][j-1]
  return velocity_commanded, velocity_actual

def add_noise_3(traj_commanded, traj_actual):
  """
  Std Dev in Noise Model proportional to the commanded velocity
  """
  K = len(traj_actual)
  velocity_commanded = np.zeros([K,N])
  velocity_actual = np.zeros([K,N])

  #Compute velocity_commanded
  for j in range(1,N):
    for i in range(K):
      velocity_commanded[i][j-1] = traj_commanded[i][j] - traj_commanded[i][j-1]
  v = LA.norm(velocity_commanded, axis=0) #Normalized velocity magnitude

  for i in range(K):
    eps = gaussian_sine_error_vel(v)
    for j in range(1,N):
      # Recurrance (prime below means actual traj with error):
      # x_{t+1}^prime = x_{t}^prime + x_{t+1} - x_{t} + eps
      traj_actual[i][j] = traj_actual[i][j-1] + traj_commanded[i][j] - traj_commanded[i][j-1] + eps[j]
      velocity_actual[i][j] = traj_actual[i][j] - traj_actual[i][j-1]
  return velocity_commanded, velocity_actual

class TrajManager(object):
  def __init__(self, fname, traj_func, 
    path_to_data, display_plot=False):
    self.fname = fname
    self.traj_func = traj_func
    self.path_to_data = path_to_data
    self.display_plot = display_plot

  def run(self):
    x,y,z = self.gen_traj()
    traj_commanded = [x,y,z]
    traj_actual = copy.deepcopy(traj_commanded)
    velocity_commanded, velocity_actual = add_noise_3(traj_commanded, traj_actual)

    traj = {}
    traj['traj_commanded'] = traj_commanded
    traj['traj_actual'] = traj_actual
    traj['velocity_commanded'] = velocity_commanded
    traj['velocity_actual'] = velocity_actual
    traj['speed_commanded'] = LA.norm(velocity_commanded, axis=0)
    traj['speed_actual'] = LA.norm(velocity_actual, axis=0)
    self.plot_traj(traj)
    self.save_traj(traj)

  def save_traj(self, traj):
    pickle.dump(traj, open(self.path_to_data + self.fname + '.p','wb'))

  def gen_traj(self):
    return self.traj_func()

  def plot_traj(self, traj):
    traj_commanded = traj["traj_commanded"]
    traj_actual = traj["traj_actual"]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(traj_commanded[0], traj_commanded[1], traj_commanded[2], 
      label=self.traj_func.__name__ + '_commanded', color='blue', linestyle='dashed')
    if traj_actual:
      ax.plot(traj_actual[0], traj_actual[1], traj_actual[2], 
        label=self.traj_func.__name__ + '_actual', color='red')
    ax.legend()
    fig.savefig(self.path_to_data+self.fname+'.png')
    if self.display_plot:
      plt.show()

def simple_helix():
  theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
  z = np.linspace(0, 3, N)
  r = 0.8
  x = r * np.sin(theta)
  y = r * np.cos(theta)
  return x,y,z

def simple_ellipse():
  theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
  z = np.linspace(0, 3, N)
  a = 0.4
  b = 0.9
  x = a * np.sin(theta)
  y = b * np.cos(theta)
  return x,y,z

def quadratic_helix():
  theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
  z = np.linspace(0, 3, N)
  r = z**2 + 1
  r = r/np.max(r)
  x = r * np.sin(theta)
  y = r * np.cos(theta)
  return x,y,z

def cubic_helix():
  theta = np.linspace(-4 * np.pi, 4 * np.pi, N)
  z = np.linspace(0, 3, N)
  r = 3*z**3
  r = r/np.max(r)
  x = r * np.sin(theta)
  y = r * np.cos(theta)
  return x,y,z

def lissajous_1():
  theta = np.linspace(-1 * np.pi, 1 * np.pi, N)
  z = np.linspace(0, 3, N)
  a = 0.4
  b = 0.9
  k_x = 3
  k_y = 2
  x = a * np.sin(k_x * theta)
  y = b * np.cos(k_y * theta)
  return x,y,z

def lissajous_2():
  theta = np.linspace(0, 1 * np.pi, N)
  z = np.linspace(0, 3, N)
  a = 0.3
  b = 0.8
  k_x = 2
  k_y = 3
  x = a * np.sin(k_x * theta)
  y = b * np.cos(k_y * theta)
  return x,y,z

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Arg parser")
  parser.add_argument('--display_plot', action='store_true', help='Display trajectory while generating them')
  parser.add_argument('--path_to_data', default='sim-data/', help='Path to saving the traj data as pickle files')
  args = parser.parse_args()
  if not os.path.exists(args.path_to_data):
    os.makedirs(args.path_to_data)
  list_funcs = [simple_helix, simple_ellipse, quadratic_helix, cubic_helix, lissajous_1, lissajous_2]
  # list_funcs = [simple_helix,]
  for fname, traj_func in enumerate(list_funcs):
    traj_manager = TrajManager(fname=str(fname), traj_func=traj_func, 
      path_to_data=args.path_to_data, display_plot=args.display_plot)
    traj_manager.run()
