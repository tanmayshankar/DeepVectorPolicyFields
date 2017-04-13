import numpy as np

def _powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  # Taken from itertools recipes in Python docs.
  from itertools import combinations, chain
  s = list(iterable)
  return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def interpolate(grid_cell_size, point):
  ''' Multilinear interpolation on a fixed grid of equal size cells '''
  base_point = grid_cell_size * np.floor(point / grid_cell_size)
  base_lengths = point - base_point
  bases = []
  for index_set in _powerset(range(len(point))):
    index_set = set(index_set)
    volume = 1
    point_to_add = base_point.copy()
    for i in range(len(point)):
      if i in index_set:
        side_length = base_lengths[i]
        point_to_add[i] += grid_cell_size
      else:
        side_length = grid_cell_size - base_lengths[i]

      volume *= side_length / grid_cell_size

    bases.append((volume, point_to_add))

  return bases

if __name__ == "__main__":
  grid_size = 0.1
  test_x = np.array((0.49, 0.59, 0.123, 32, 1.2))
  split = interpolate(grid_size, test_x)

  def f(x, y, z, a, b):
    return 2 * x**3 + 3 * y ** 2 - z  + a * b - z * b

  recovered = test_x * 0
  interp_value = 0
  for percent, point in split:
    recovered += percent * point
    interp_value += percent * f(*point)

  from scipy.interpolate import RegularGridInterpolator
  x = np.arange(0, 1, grid_size)
  y = np.arange(0, 1, grid_size)
  z = np.arange(0, 1, grid_size)
  a = np.arange(31, 33, grid_size)
  b = np.arange(1, 2, grid_size)
  data = f(*np.meshgrid(x, y, z, a, b, indexing='ij', sparse=True))
  interp_f = RegularGridInterpolator((x, y, z, a, b), data)

  print("input", test_x)
  print("recovered", recovered)
  print("interpolated", interp_value)
  print("scipy says", interp_f(test_x))
