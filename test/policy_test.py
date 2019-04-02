import os

import numpy as np
import torch

FILE_PREFIX = os.path.expanduser('~/path/mv/')
file_name = 'model/policy_best.m'
model = torch.load(FILE_PREFIX +file_name)

state = np.array([2,-1,-1,-1,-1])
a = model.select_action(np.array(state), None)

print(a)
