from itertools import count
import torch
import numpy as np



if __name__ == "__main__":
  a = np.array([1, -1, -1, -1])
  b = np.array([1, 1, 0, 0])

  result = a > 0
  print(sum((result == b)))