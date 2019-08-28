import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import NCF
import random
import time


def build_dictionaries():
    ratings = open('ratings.csv', 'r').readlines()

    ratings = ratings[1:]
    print(ratings[:4])

    user_to_books = {}          # {user: {book0: rating, book1: rating, ...}}
    book_to_users = {}          # {book: {user0: rating, user1: rating, ...}}

    for line in ratings:
        user_id, book_id, rating = line.split(',')
        rating = int(rating)

        if user_id in user_to_books.keys():
            # this assumes there aren't multiple ratings for a user-book pair, which there shouldn't
            # might be better to check first though
            user_to_books[user_id][book_id] = rating
        else:
            user_to_books[user_id] = {book_id: rating}

        if book_id in book_to_users.keys():
            book_to_users[book_id][user_id] = rating
        else:
            book_to_users[book_id] = {user_id: rating}

    user_num = len(user_to_books.keys())
    book_num = len(book_to_users.keys())
    print(user_num, 'users', book_num, 'books')

    return user_to_books, book_to_users


user_to_books, book_to_users = build_dictionaries()

all_users = list(user_to_books.keys())
all_books = list(book_to_users.keys())

pos_user_books = open('user_book.csv').readlines()
random.shuffle(pos_user_books)

# converting to ints
for i in range(len(pos_user_books)):
    line = pos_user_books[i].split(',')
    pos_user_books[i] = [int(line[0]), int(line[1])]

def get_neg_user_books():
    book = random.choice(all_books)
    user = random.choice(all_users)
    while user in list(book_to_users[book].keys()):
        user = random.choice(all_users)
    return [int(user), int(book)]


def get_batch(size=50):
    pos_num = int(1/4*size)         # 3 <= neg:pos <= 6  as described by section 4.3 in paper
    i = random.randint(0, len(all_books)-pos_num-1)
    pos = pos_user_books[i:i+pos_num]
    neg_num = size - pos_num
    neg = [get_neg_user_books() for i in range(neg_num)]
    target = torch.cat((torch.ones(pos_num, 1), torch.zeros(neg_num, 1)), dim=0)
    return torch.Tensor(pos+neg).long().cuda(), target.cuda()


def train(mode, optimizer):
    # TODO use early stopping
    for i in range(int(1e4)):
        x, y = get_batch()
        y_ = ncf(x, mode=mode)
        loss = F.binary_cross_entropy(y_, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print('step', i, 'loss', loss.item())


ncf = NCF.NeuralCollaborativeFiltering(len(all_users)+1, len(all_books)+1, 100, 100, 128).cuda()

mlp_optimizer = optim.Adam(list(ncf.mlp_item_embeddings.parameters()) +
                        list(ncf.mlp_user_embeddings.parameters()) +
                        list(ncf.mlp.parameters()) +
                        list(ncf.mlp_out.parameters()), lr=1e-4)
gmf_optimizer = optim.Adam(list(ncf.gmf_item_embeddings.parameters()) +
                        list(ncf.gmf_user_embeddings.parameters()) +
                        list(ncf.gmf_out.parameters()), lr=1e-4)
ncf_optimizer = optim.SGD(ncf.parameters(), lr=1e-4)

print('\nTraining MLP')
train('mlp', mlp_optimizer)

print('\nTrainging GMF')
train('gmf', gmf_optimizer)

ncf.join_output_weights()

print('\nTraining NCF')
train('ncf', ncf_optimizer)
