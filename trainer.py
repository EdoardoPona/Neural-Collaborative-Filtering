import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import NCF
import random
import time
import movielens_loader
import numpy as np


dataset_loader = movielens_loader.MovielensDatasetLoader()

user2items, item2users, test_user2items = dataset_loader.build_dictionaries()
user_num, item_num = dataset_loader.user_num, dataset_loader.item_num


def get_pos_user():
    """ removes the element every time, so that we know what data we have trained the model on """
    user = random.choice(list(user2items.keys()))
    items = list(user2items[user].keys())
    i = random.randint(0, len(items)-1)
    # item = random.choice(list(user2items[user].keys()))
    del user2items[user][items[i]]
    if len(user2items[user]) == 0:
        del user2items[user]
    return [int(user), int(items[i])]


def get_neg_user():
    item = random.choice(list(item2users.keys()))
    user = random.choice(list(user2items.keys()))
    while user in list(item2users[item].keys()):
        user = random.choice(list(user2items.keys()))
    return [int(user), int(item)]


def get_batch(size=128):
    pos_num = int(1/5*size)
    pos = [get_pos_user() for i in range(pos_num)]
    neg = [get_neg_user() for i in range(size - pos_num)]
    target = torch.cat((torch.ones(pos_num), torch.zeros(size-pos_num)), dim=0)
    return torch.Tensor(pos+neg).long().cuda(), target.cuda()


def HitRatio(test_num=100):
    global test_user2items
    hits = 0
    test_users = list(test_user2items.keys())[:test_num]
    for user in test_users:
        x = [get_neg_user() for i in range(99)]
        x.append([int(user), int(test_user2items[user])])
        x_id = len(x)-1         # index of the last item, the positive one
        x = torch.Tensor(x).long().cuda()
        y = ncf(x, mode='ncf').squeeze(1).cpu().detach().numpy()
        sort_ids = np.argsort(y)

        if x_id in sort_ids[-10:]:
            hits += 1
    return hits / test_num


def train(mode, optimizer, epochs=10, batch_size=128):
    # TODO use early stopping
    global user2items
    global item2users
    global test_user2items
    ep = 0
    i = 0
    while ep < epochs:
        x, y = get_batch(128)
        y_ = ncf(x, mode=mode)
        loss = F.binary_cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print('epoch', ep, 'step', i, 'loss', loss.item())
        i += 1

        if i % 5000 == 0:
            print('hit ratio', HitRatio())

        if len(user2items) < batch_size:    # really there might be more left, but this is the minimum guaranteed
            ep += 1
            # user2items, item2users = dataset_loader.build_dictionaries()
            user2items, item2users, test_user2items = dataset_loader.get_dictionaries()


ncf = NCF.NeuralCollaborativeFiltering(user_num+1, item_num+1, 16).cuda()
ncf.join_output_weights()
print('Hit ratio:', HitRatio())

mlp_optimizer = optim.Adam(list(ncf.mlp_item_embeddings.parameters()) +
                        list(ncf.mlp_user_embeddings.parameters()) +
                        list(ncf.mlp.parameters()) +
                        list(ncf.mlp_out.parameters()), lr=1e-3)
gmf_optimizer = optim.Adam(list(ncf.gmf_item_embeddings.parameters()) +
                        list(ncf.gmf_user_embeddings.parameters()) +
                        list(ncf.gmf_out.parameters()), lr=1e-3)
ncf_optimizer = optim.Adam(ncf.parameters(), lr=5e-4)


print('\nTraining MLP')
train('mlp', mlp_optimizer, epochs=1)

print('\nTrainging GMF')
train('gmf', gmf_optimizer, epochs=1)

ncf.join_output_weights()

print('\nTraining NCF')
train('ncf', ncf_optimizer, epochs=20)
