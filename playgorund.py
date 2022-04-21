# Python 3 program to demonstrate torch.stack() method
# for two 2D tensors.
# importing torch
import torch

# creating tensors
x = torch.tensor([[1., 3., 6.], [10., 13., 20.], [10., 13., 20.]])
y = torch.tensor([[2., 7., 9.], [14., 21., 34.], [10., 13., 20.]])

# printing above created tensors
print("Tensor x:\n", x)
print("Tensor y:\n", y)

# join above tensor using "torch.stack()"
print("join tensors")
t = torch.stack((x, y))

# print final tensor after join
print(t[0: :])
print(torch.max(torch.tensor(1.), torch.tensor(0.)))



