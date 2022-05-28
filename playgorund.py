from itertools import count
import torch
import numpy as np
import torch.nn as nn


if __name__ == "__main__":
  # m = nn.Sigmoid() # initialize sigmoid layer
  # loss = nn.BCELoss() # initialize loss function
  # input = torch.randn(3, requires_grad=True) # give some random input
  # print(input)
  # target = torch.empty(3).random_(2)
  # print(target) # create some ground truth values
  # print(m(input))
  # output = loss(m(input), target) # forward pass
  # print(output)
  # output.backward() # backward pass
  # s = "person_66\p66_L_20.jpeg"
  # person_number = s.split("\\")[0].split("_")[1]
  # print(person_number)
  arabic = r"C:\Users\FinalProject\Desktop\backup_models\CrossEntropy\lr_0003\a.txt"
  model_bylang = {"Arabic" : arabic }
  print(model_bylang["Arabic"])