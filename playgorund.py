




from time import sleep
from traceback import print_tb


list_of_lists = [[1] for i in range(10)]
list_of_lists1 = [[1]] * 10 

list_of_lists[0][0] = 10
list_of_lists1[0][0] = 10

print(list_of_lists)
print(list_of_lists1)