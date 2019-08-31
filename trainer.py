import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import NCF
import random
import time
import movielens_loader


dataset_loader = movielens_loader.MovielensDatasetLoader()
user2items, item2users = dataset_loader.build_dictionaries()
user_num, item_num = dataset_loader.user_num, dataset_loader.item_num


def get_pos_user():
    user = random.choice(list(user2items.keys()))
    item = random.choice(list(user2items[user].keys()))
    return [int(user), int(item)]


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


def train(mode, optimizer):
    # TODO use early stopping
    for i in range(int(1e5)):
        x, y = get_batch()
        y_ = ncf(x, mode=mode)
        loss = F.binary_cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print('step', i, 'loss', loss.item())


ncf = NCF.NeuralCollaborativeFiltering(user_num+1, item_num+1, 64).cuda()

mlp_optimizer = optim.Adam(list(ncf.mlp_item_embeddings.parameters()) +
                        list(ncf.mlp_user_embeddings.parameters()) +
                        list(ncf.mlp.parameters()) +
                        list(ncf.mlp_out.parameters()), lr=1e-3)
gmf_optimizer = optim.Adam(list(ncf.gmf_item_embeddings.parameters()) +
                        list(ncf.gmf_user_embeddings.parameters()) +
                        list(ncf.gmf_out.parameters()), lr=1e-3)
ncf_optimizer = optim.SGD(ncf.parameters(), lr=5e-4)

print('\nTraining MLP')
train('mlp', mlp_optimizer)

print('\nTrainging GMF')
train('gmf', gmf_optimizer)

ncf.join_output_weights()

print('\nTraining NCF')
train('ncf', ncf_optimizer)

