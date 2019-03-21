import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.externals import joblib
from torch.distributions import Categorical


gamma = 0.99


class Policy(nn.Module):
    def  __init__(self, input_size, output_size, hidden_size=20):
        super(Policy, self).__init__()
        # state -> hidden
        self.affine1 = nn.Linear(input_size, 20)
        # hidden -> action
        self.affine2 = nn.Linear(20, output_size)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x, train=False):
        if train:
            model = torch.nn.Sequential(
                self.affine1,
                # TODO 需要检验失活函数是否需要
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                self.affine2,
                nn.ReLU(),
                # 使用crossentropyloss 不需要softmax层
                # nn.Softmax(dim=1)
            )
        else:
            model = torch.nn.Sequential(
                self.affine1,
                # TODO 需要检验失活函数是否需要
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                self.affine2,
                nn.ReLU(),
                # 使用crossentropyloss 不需要softmax层
                nn.Softmax()
            )
        return model(x)

    def select_action(self,state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self(state)
        # print('probs',probs.tolist())
        m = Categorical(probs)
        action = m.sample()
        # t = m.log_prob(action)
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def select_best_action(self,state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self(state)
        action = torch.argmax(probs,dim=1).tolist()
        return action[0]

    def update_policy(self, optimizer):
        # print('update starts')
        R = 0
        policy_loss = []
        rewards = []
        # print('reward_before',self.rewards)
        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        # std = rewards.std()
        if len(rewards) == 1:
            # TODO if rewards length is one , do nothing
            pass
        else:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        # print('reward:', rewards)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        # print('policy_loss', policy_loss)
        # print(policy_loss)
        policy_loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def add_reward(self, reward):
        self.rewards.append(reward)


if __name__ == '__main__':
    state1 = np.array([-1,-1,-1,-1, -1])
    state2 = np.array([57,-1,-1,-1, -1])
    state3 = np.array([-1,-1,-1,-1, -1])
    # state = torch.from_numpy(state).float().unsqueeze(0)

    FILE_PREFIX = '/path/mv/model/'
    file_name = 'policy_pretrain_0.7437.pkl'
    model = torch.load(FILE_PREFIX + file_name)

    a = torch.tensor(state1).float()
    b = a.std()
    mean = a.mean()

    for i in range(300):
        action1 = model.select_action(np.array(state1))
        print('action1', action1)
        model.rewards.append(-1)
        action2 = model.select_action(np.array(state2))
        print('action2', action2)
        model.rewards.append(-30)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

        model.update_policy(optimizer)





