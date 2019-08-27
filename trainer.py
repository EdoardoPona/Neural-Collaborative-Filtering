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
    return [int(book), int(user)]


def get_batch(size=50):
    pos_num = int(2/5*size)         # TODO justify/change this number
    i = random.randint(0, len(all_books)-pos_num-1)
    pos = pos_user_books[i:i+pos_num]
    neg_num = size - pos_num
    neg = [get_neg_user_books() for i in range(neg_num)]
    target = torch.cat((torch.ones(pos_num, 1), torch.zeros(neg_num, 1)), dim=0)
    return torch.Tensor(pos+neg), target

def pretrain_gmf():
    # train only gmf, use early stopping with Adam
    pass

def pretrain_mlp():
    # train only mlp, use early stopping with Adam
    pass


def train():
    pretrain_gmf()
    pretrain_mlp()
    # join output layer weights
    # train the whole model with SGD
