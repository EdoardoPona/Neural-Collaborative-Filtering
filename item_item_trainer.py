import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import NCF
import random
import time
import movielens_loader
import numpy as np
import copy

dataset_loader = movielens_loader.MovielensDatasetLoader(mode='item_item')

user2items, item2users, test_item_pair = dataset_loader.build_dictionaries()

# used for picking the negative pairs, users should not be removed from here
all_users = copy.deepcopy(list(user2items.keys()))
all_user2items = copy.deepcopy(user2items)

user_num, item_num = dataset_loader.user_num, dataset_loader.item_num


def get_pos_pair():
    """ removes the element every time, so that we know what data we have trained the model on """
    user = random.choice(list(user2items.keys()))
    item_keys = list(user2items[user].keys())

    id = random.randint(0, len(item_keys)-1)
    item0 = item_keys[id]
    del user2items[user][item_keys[id]]
    del item_keys[id]

    id = random.randint(0, len(item_keys)-1)
    item1 = item_keys[id]
    del user2items[user][item_keys[id]]
    del item_keys[id]

    if len(user2items[user]) < 2:               # we need at least a pair for each user left
        del user2items[user]

    return [int(item0), int(item1)]


def get_neg_pair():
    item0 = random.choice(list(item2users.keys()))      # first item randomly picked
    # the user of the second item should not have rated the first item, so it should not be in item2users[item0]
    item1_user = random.choice(all_users)     # random out of all users
    while item1_user in list(item2users[item0].keys()):         # the random user has rated item0
        item1_user = random.choice(all_users)  # pick another one

    item1 = random.choice(list(all_user2items[item1_user].keys()))      # pick random item out of the ones rated by item1_user
    return [int(item0), int(item1)]


def get_batch(size=128):
    pos_num = int(1/5*size)
    pos = [get_pos_pair() for i in range(pos_num)]
    neg = [get_neg_pair() for i in range(size - pos_num)]
    target = torch.cat((torch.ones(pos_num), torch.zeros(size-pos_num)), dim=0)
    return torch.Tensor(pos+neg).long().cuda(), target.cuda()


def HitRatio(test_num=200):
    global test_item_pair
    hits = 0
    test_users = list(test_item_pair.keys())[:test_num]
    for user in test_users:
        x = [get_neg_pair() for i in range(99)]
        x.append(test_item_pair[user])
        x_id = len(x)-1
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
    global test_item_pair
    ep = 0
    i = 0
    while ep < epochs:
        x, y = get_batch(128)
        y_ = ncf(x, mode=mode)
        loss = F.binary_cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 400 == 0:
            print('epoch', ep, 'step', i, 'loss', loss.item())
        i += 1

        if i % 5000 == 0:
            print('hit ratio', HitRatio())

        if len(user2items) < batch_size:    # really there might be more left, but this is the minimum guaranteed
            ep += 1
            # user2items, item2users = dataset_loader.build_dictionaries()
            user2items, item2users, test_item_pair = dataset_loader.get_dictionaries()


ncf = NCF.NCF_item_item(item_num+1, 16).cuda()
ncf.join_output_weights()
print('Hit ratio:', HitRatio())

mlp_optimizer = optim.Adam(list(ncf.mlp_item_embeddings.parameters()) +
                        list(ncf.mlp.parameters()) +
                        list(ncf.mlp_out.parameters()), lr=1e-3)
gmf_optimizer = optim.Adam(list(ncf.gmf_item_embeddings.parameters()) +
                        list(ncf.gmf_out.parameters()), lr=1e-3)
ncf_optimizer = optim.Adam(ncf.parameters(), lr=5e-4)


print('\nTraining MLP')
train('mlp', mlp_optimizer, epochs=1)

print('\nTrainging GMF')
train('gmf', gmf_optimizer, epochs=1)

ncf.join_output_weights()

print('\nTraining NCF')
train('ncf', ncf_optimizer, epochs=20)
