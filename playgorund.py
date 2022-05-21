from itertools import count
import torch
import numpy as np
import torch.nn as nn


if __name__ == "__main__":
  m = nn.Sigmoid() # initialize sigmoid layer
  loss = nn.BCELoss() # initialize loss function
  input = torch.randn(3, requires_grad=True) # give some random input
  print(input)
  target = torch.empty(3).random_(2)
  print(target) # create some ground truth values
  print(m(input))
  output = loss(m(input), target) # forward pass
  print(output)
  output.backward() # backward pass

