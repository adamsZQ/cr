import os
import sys

import h5py

import numpy as np
import torch
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as Data

from policy.policy_v2 import Policy

data_file = '/path/mv/pretrain_data_5turns.h5'
f = h5py.File(data_file, 'r')
states = f['states'][:]
actions = f['actions'][:]
actions = np.squeeze(actions,axis=1)
# actions_tc_list = []
# for action in actions:
#     actions_tc = [0] * (np.max(actions)+1)
#     actions_tc[action] = 1
#     actions_tc_list.append(actions_tc)

actions_tc_list = np.array(actions)

input_size = states.shape[1]
output_size = np.max(actions_tc_list) + 1

# inputs = torch.from_numpy(states).float()
# target = torch.from_numpy(actions_tc_list).float()

X_train, X_test, y_train, y_test = train_test_split(states, actions_tc_list, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
X_val = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
y_val = torch.from_numpy(y_val).long()


# 添加批训练
batch_size = 32
torch_dataset = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

policy = Policy(input_size, output_size)
criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


def train(model_prefix, file_name):
    num_epochs = 100000
    best_loss = 1.6
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            # forward
            out = policy(batch_x)  # 前向传播
            loss = criterion(out, batch_y)  # 计算loss

            # backward
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        if epoch % 1 == 0:
            val_loss = val(X_val,y_val)
            print('Epoch[{}/{}]'.format(epoch, num_epochs) + 'loss: {:.6f}'.format(
                loss.item()) + 'val_loss:{:.6f}'.format(val_loss))
            sys.stdout.flush()

            best_loss = save_model(policy, model_prefix, file_name, val_loss, best_loss)


def val(X_val, y_val):
    y_pred= policy(X_val)
    val_loss = criterion(y_pred, y_val)  # 计算loss
    return val_loss.item()


def test(X_test, y_test):
    y_pred = policy(X_test)
    loss = criterion(y_pred, y_test)  # 计算loss
    print(loss.item())


def save_model(model, file_prefix=None, file_name=None, val_loss='None', best_loss='None', enforcement = False):
    # Save model
    try:
        if enforcement or val_loss == 'None' or best_loss == 'None':
            file_path = '{}{}_{}.pkl'.format(file_prefix, file_name, 'enforcement')
            torch.save(model, file_path)
            print('enforcement save:', file_path)

        elif val_loss != 'None' and best_loss != 'None' and ~enforcement:
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            if is_best:
                file_path = '{}{}_{:.4f}.pkl'.format(file_prefix, file_name, best_loss)
                torch.save(model, file_path)
            return best_loss
    except Exception as e:
        # if error, save model in default path
        print(e)
        file_path = '{}{}.pkl'.format(os.getcwd(), '/default')
        print('default save:', file_path)
        torch.save(model, file_path)


if __name__ == '__main__':
    FILE_PREFIX = '/path/mv/model/'
    file_name = '5turns/policy_pretrain_0.7437.pkl'
    train(FILE_PREFIX, file_name)

    # policy = torch.load(FILE_PREFIX+file_name)
    # test(X_test, y_test)
    '''
    pretrain result:val_loss:0.7437, test_loss:0.7498
    '''



