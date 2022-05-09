from re import S
import torch
thresh=1


def a():
  global thresh
  thresh+= 1
  print(thresh)
  


if __name__ == "__main__":

  ss="person_242/p242_a_L_6.jpeg"
  a = "b" in ss
  print(a)
