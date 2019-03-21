import collections
import torch

import numpy as np

# a = np.argsort([1414,22222,31],)
#
# index_downsort = a[::-1]
#
# lll = index_downsort[:5]
# print(lll)
#
#
# print('aa{}'.format(a))
#
# item_sort = [1,4,51,1]
# predict = [1.1,2.2,3.3,4.5]
# a = zip(item_sort, predict)
# print(list(a))

# x = [1,64,3]
#
# y = [213,235,1]
#
# if set(x) & set(y):
#     print(True)


# genres = {'a':1, 'b':2}
#
# # a = genres.pop('a')
#
# print('b' in genres)

#
# a = ['1', '2', '3']
#
# dict = {set(a):1}

# print(set([1,2]).issubset(set([2,1])))]

# x = []
# x.append([0]*10)
# x.append([1]*10)
# x = np.array(x).reshape(-1,5)
# print(x)

# x = []
# x.append(set([1,2,3]))
# x.append(set([3,2]))
#
# b = x.index(set([2,3]))
# print(b)

b = torch.randn(3)
print(b)
print(torch.max(b,0)[1].tolist())