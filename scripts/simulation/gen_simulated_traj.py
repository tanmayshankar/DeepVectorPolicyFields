import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from mpl_toolkits.mplot3d import Axes3D

N = 1000
mu_noise = 0.0
sigma_noise = 0.001

def add_noise(x,y,z):
  x = x + np.random.normal(mu_noise, sigma_noise, N)
  y = y + np.random.normal(mu_noise, sigma_noise, N)
  z = z + np.random.normal(mu_noise, sigma_noise, N)
  return x, y, z

class TrajManager(object):
  def __init__(self, fname, traj_func, 
    path_to_data, display_plot=False):
    self.fname = fname
    self.traj_func = traj_func
    self.path_to_data = path_to_data
    self.display_plot = display_plot

  def run(self):
    x,y,z = self.gen_traj()
    x,y,z = add_noise(x,y,z)
    self.plot_traj(x,y,z)
    self.save_traj(x,y,z)

  def save_traj(self,x,y,z):
    traj = np.array([x,y,z]).T
    pickle.dump(traj, open(self.path_to_data + self.fname + '.p','wb'))

  def gen_traj(self):
    return self.traj_func()

  def plot_traj(self, x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, label=self.traj_func.func_name)
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
  # list_funcs = [simple_helix, simple_ellipse, quadratic_helix, cubic_helix, lissajous]
  list_funcs = [simple_helix, simple_ellipse, cubic_helix, lissajous_1, lissajous_2]
  # list_funcs = [simple_ellipse]
  for fname, traj_func in enumerate(list_funcs):
    traj_manager = TrajManager(fname=str(fname), traj_func=traj_func, 
      path_to_data=args.path_to_data, display_plot=args.display_plot)
    traj_manager.run()